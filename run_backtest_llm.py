#!/usr/bin/env python3
"""冷启动SKU回测：用 LLM (Claude) 通过同品类数据预测新SKU销量

思路：
  - 给大模型提供：目标SKU自身历史 + 同品类成熟SKU同期数据 + 未来协变量 + 自动生成的运营备注
  - 7天滚动预测，history_end固定不推进，避免测试期数据泄露
  - 协变量：广告预算、促销、折扣率、节假日等（不含未来不可知的用户行为数据）
  - 额外信息：去年同期品类销量（季节性参考）、自动生成的运营备注（增长趋势、广告变化等）

使用方法：
  1. 配置 AWS Bedrock 访问权限
  2. 将数据文件放到 data/uploads/ 目录
  3. 修改下方 DATA_FILE / REGION / MODEL_ID 配置
  4. python3 run_backtest_llm.py
"""
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import json
import time
import re
import boto3

# ===== 配置 =====
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'
DATA_FILE = DATA_DIR + '/uploads/销售脱敏数据_v2.xlsx'  # 替换为你的数据文件
OUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/results'
ROLL_DAYS = 7
REGION = os.environ.get('AWS_REGION', 'us-east-1')
MODEL_ID = os.environ.get('LLM_MODEL_ID', 'us.anthropic.claude-opus-4-6-v1')

# 训练/测试期分割
TRAIN_END = '2025-09-30'
TEST_START = '2025-10-01'
TEST_END = '2025-11-29'

# 冷启动阈值（训练期非零天数低于此值视为冷启动，默认90天）
COLD_START_THRESHOLD = int(os.environ.get('COLD_START_DAYS', '90'))
# 成熟SKU阈值（同品类参考SKU需达到此天数）
MATURE_THRESHOLD = 180

# 美国节假日/大促（测试期内）
US_EVENTS = {
    '2025-10-07': 'Prime Big Deal Days Day1',
    '2025-10-08': 'Prime Big Deal Days Day2',
    '2025-10-13': 'Columbus Day',
    '2025-10-31': 'Halloween',
    '2025-11-11': 'Veterans Day',
    '2025-11-22': 'Pre-Black Friday deals start',
    '2025-11-27': 'Thanksgiving',
    '2025-11-28': 'Black Friday',
    '2025-11-29': 'Small Business Saturday',
}

# 聚合字段
SUM_COLS = ['quantity', 'ppc_fee', 'ppc_clicks', 'ppc_ad_order_quantity',
            'sessions', 'ppc_impression']
MEAN_COLS = ['discount_rate', 'conversion_rate']
MIN_COLS = ['大类排名', '小类排名']


def load_data():
    """加载并按SKU+日期聚合"""
    raw = pd.read_excel(DATA_FILE)
    raw['date'] = pd.to_datetime(raw['purchase_date'], format='%Y%m%d')

    agg_dict = {c: 'sum' for c in SUM_COLS if c in raw.columns}
    agg_dict.update({c: 'mean' for c in MEAN_COLS if c in raw.columns})
    agg_dict.update({c: 'min' for c in MIN_COLS if c in raw.columns})
    agg_dict['品类'] = 'first'
    agg_dict['title'] = 'first'
    if 'avg_deal_fee' in raw.columns:
        agg_dict['avg_deal_fee'] = 'mean'

    daily = raw.groupby(['sku', 'date']).agg(agg_dict).reset_index()
    return daily.sort_values(['sku', 'date'])


def get_cold_start_skus(daily):
    """找出冷启动SKU（训练期<阈值天非零 + 测试期>=10天非零 + 有同品类成熟SKU）"""
    results = []
    for sku in daily['sku'].unique():
        s = daily[daily['sku'] == sku]
        cat = s['品类'].iloc[0]
        train = s[s['date'] <= TRAIN_END]
        test = s[(s['date'] >= TEST_START) & (s['date'] <= TEST_END)]
        train_nz = (train['quantity'] > 0).sum()
        test_nz = (test['quantity'] > 0).sum()

        if train_nz >= COLD_START_THRESHOLD or test_nz < 10:
            continue

        peers = daily[(daily['品类'] == cat) & (daily['sku'] != sku)]['sku'].unique()
        mature_peers = [p for p in peers
                        if (daily[(daily['sku'] == p) & (daily['date'] <= TRAIN_END)]['quantity'] > 0).sum() >= MATURE_THRESHOLD]

        if mature_peers:
            results.append({
                'sku': sku, 'cat': cat, 'title': s['title'].iloc[0],
                'train_nz': train_nz, 'test_nz': test_nz,
                'peers': mature_peers,
            })
    return sorted(results, key=lambda x: len(x['peers']), reverse=True)


def generate_notes(daily, sku, peers, history_end):
    """从数据自动生成运营备注"""
    s = daily[(daily['sku'] == sku) & (daily['date'] <= history_end) & (daily['quantity'] > 0)].sort_values('date')
    notes = []

    if len(s) > 0:
        first = s['date'].min()
        days_active = (history_end - first).days
        notes.append(f'该SKU首次有销量日期为{first.strftime("%Y-%m-%d")}，已上架{days_active}天，有{len(s)}天有销量记录。')

    if len(s) >= 28:
        last14 = s.tail(14)['quantity'].mean()
        prev14 = s.tail(28).head(14)['quantity'].mean()
        if prev14 > 0:
            growth = (last14 - prev14) / prev14 * 100
            trend = '快速增长' if growth > 30 else '增长' if growth > 10 else '平稳' if growth > -10 else '下降'
            notes.append(f'近期趋势：最近14天日均销量{last14:.0f}，前14天日均{prev14:.0f}，{trend}（{growth:+.0f}%）。')
    elif len(s) > 0:
        notes.append(f'数据较少（仅{len(s)}天有销量），最近日均销量{s.tail(7)["quantity"].mean():.0f}。')

    if len(s) >= 14:
        recent_ppc = s.tail(7)['ppc_fee'].mean()
        prev_ppc = s.tail(14).head(7)['ppc_fee'].mean()
        if prev_ppc > 0:
            ppc_change = (recent_ppc - prev_ppc) / prev_ppc * 100
            if abs(ppc_change) > 20:
                notes.append(f'广告投放：最近7天日均广告费{recent_ppc:.0f}，较前7天{"增加" if ppc_change > 0 else "减少"}{abs(ppc_change):.0f}%。')

    ranks = s[s['小类排名'].notna()]
    if len(ranks) >= 14:
        r7, p7 = ranks.tail(7), ranks.tail(14).head(7)
        sn, sp = r7['小类排名'].mean(), p7['小类排名'].mean()
        bn, bp = r7['大类排名'].mean(), p7['大类排名'].mean()
        notes.append(f'排名趋势：小类第{sp:.0f}→第{sn:.0f}名（{"上升" if sn < sp else "下降"}），大类第{bp:.0f}→第{bn:.0f}名（{"上升" if bn < bp else "下降"}）。')
    elif len(ranks) > 0:
        notes.append(f'最近小类排名第{ranks.tail(7)["小类排名"].mean():.0f}名，大类排名第{ranks.tail(7)["大类排名"].mean():.0f}名。')

    if 'avg_deal_fee' in s.columns and len(s) >= 7:
        price = s.tail(7)['avg_deal_fee'].mean()
        if price > 0:
            notes.append(f'近期平均售价约{price:.1f}。')

    peer_means = []
    for p in peers:
        ps = daily[(daily['sku'] == p) & (daily['date'] <= history_end)]
        if len(ps) > 0:
            peer_means.append((p, ps.tail(14)['quantity'].mean()))
    if peer_means:
        peer_means.sort(key=lambda x: x[1], reverse=True)
        notes.append(f'同品类成熟SKU近14天日均销量：{", ".join(f"{p}日均{m:.0f}" for p, m in peer_means)}。')

    if len(s) >= 7:
        cvr = s.tail(7)['conversion_rate'].mean()
        if cvr > 0:
            notes.append(f'最近7天平均转化率{cvr:.2%}。')

    return '\n'.join(notes)


def format_sku_history(daily, sku, start_date, end_date, max_days=30):
    """格式化SKU历史数据为CSV文本"""
    s = daily[(daily['sku'] == sku) & (daily['date'] >= start_date) & (daily['date'] <= end_date)].tail(max_days)
    if len(s) == 0:
        return "无数据"

    lines = ["日期,销量,sessions,广告费,广告点击,广告订单,折扣率,转化率,大类排名,小类排名"]
    for _, r in s.iterrows():
        event = US_EVENTS.get(r['date'].strftime('%Y-%m-%d'), '')
        ev = f" [{event}]" if event else ""
        big_r = int(r['大类排名']) if pd.notna(r.get('大类排名')) else 'N/A'
        small_r = int(r['小类排名']) if pd.notna(r.get('小类排名')) else 'N/A'
        lines.append(
            f"{r['date'].strftime('%Y-%m-%d')},{int(r['quantity'])},"
            f"{int(r['sessions'])},{r['ppc_fee']:.0f},{int(r['ppc_clicks'])},"
            f"{int(r['ppc_ad_order_quantity'])},{r['discount_rate']:.4f},"
            f"{r['conversion_rate']:.4f},{big_r},{small_r}{ev}")
    return "\n".join(lines)


def format_future_covariates(daily, sku, dates):
    """格式化未来已知协变量 — 排名是结果指标，不可作为未来协变量"""
    lines = ["日期,广告预算,折扣率,星期几,节假日/大促"]
    for d in dates:
        row = daily[(daily['sku'] == sku) & (daily['date'] == d)]
        ppc = row['ppc_fee'].values[0] if len(row) > 0 else 0
        disc = row['discount_rate'].values[0] if len(row) > 0 else 0
        event = US_EVENTS.get(d.strftime('%Y-%m-%d'), '无')
        lines.append(f"{d.strftime('%Y-%m-%d')},{ppc:.0f},{disc:.4f},{d.strftime('%A')},{event}")
    return "\n".join(lines)


def format_yoy_data(daily, cat, pred_dates, peers):
    """格式化去年同期的品类销量数据（季节性参考）"""
    yoy_start = pred_dates[0] - pd.DateOffset(years=1) - pd.Timedelta(days=7)
    yoy_end = pred_dates[-1] - pd.DateOffset(years=1)

    cat_data = daily[(daily['品类'] == cat) & (daily['date'] >= yoy_start) &
                     (daily['date'] <= yoy_end) & (daily['sku'].isin(peers))]
    if len(cat_data) == 0:
        return "无去年同期数据"

    cat_daily = cat_data.groupby('date').agg(
        total_qty=('quantity', 'sum'), avg_qty=('quantity', 'mean'),
        sku_count=('sku', 'nunique'),
    ).reset_index().sort_values('date')

    lines = ["日期,品类总销量,SKU平均销量,活跃SKU数"]
    for _, r in cat_daily.iterrows():
        today_equiv = (r['date'] + pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        event = US_EVENTS.get(today_equiv, '')
        ev = f" [今年对应{today_equiv}{' ' + event if event else ''}]"
        lines.append(f"{r['date'].strftime('%Y-%m-%d')},{int(r['total_qty'])},{r['avg_qty']:.0f},{int(r['sku_count'])}{ev}")
    return "\n".join(lines)


def build_prompt(daily, target_sku, target_info, pred_dates, history_end):
    """构建完整的预测prompt"""
    cat, title, peers = target_info['cat'], target_info['title'], target_info['peers']
    hist_start = history_end - pd.Timedelta(days=45)

    target_hist = format_sku_history(daily, target_sku, hist_start, history_end)
    sku_notes = generate_notes(daily, target_sku, peers, history_end)

    # 同品类成熟SKU（取数据量最多的top 3）
    top_peers = sorted(peers, key=lambda p: (daily[(daily['sku'] == p) & (daily['date'] <= history_end) & (daily['quantity'] > 0)].shape[0]), reverse=True)[:3]

    peer_sections = []
    for p in top_peers:
        p_title = daily[daily['sku'] == p]['title'].iloc[0]
        p_hist = format_sku_history(daily, p, hist_start, history_end, 30)
        p_future = format_sku_history(daily, p, pred_dates[0], pred_dates[-1])
        p_notes = generate_notes(daily, p, [], history_end)
        peer_sections.append(
            f"### {p} ({str(p_title)[:60]})\n备注: {p_notes}\n\n"
            f"历史数据（截至{history_end.strftime('%Y-%m-%d')}）:\n{p_hist}\n\n"
            f"预测期实际数据（品类趋势参考）:\n{p_future}")

    yoy_text = format_yoy_data(daily, cat, pred_dates, peers)
    future_cov = format_future_covariates(daily, target_sku, pred_dates)

    return f"""你是一个电商销量预测专家。请根据以下信息，预测目标SKU未来{len(pred_dates)}天的每日销量。

## 目标SKU信息
- SKU: {target_sku}
- 品类: {cat}
- 商品: {str(title)[:100] if pd.notna(title) else 'N/A'}

## 运营备注
{sku_notes}

## 目标SKU历史数据
{target_hist}

## 同品类成熟SKU数据（用于借鉴品类趋势和季节性模式）
{chr(10).join(peer_sections)}

## 去年同期品类销量（季节性参考）
{yoy_text}

## 未来已知信息（预测期）
{future_cov}

## 预测要求
1. 综合考虑：目标SKU自身趋势、同品类SKU的销量变化模式、广告预算变化、折扣率、节假日/大促影响
2. 注意品类季节性：同品类SKU在预测期的销量变化反映了品类整体趋势
3. 参考去年同期数据判断季节性涨跌幅度，但注意今年品类可能整体增长或下降
4. 注意大促效应：Prime Day、Black Friday等大促会导致销量激增，参考同品类SKU在大促期间的销量倍率
5. 新SKU的销量规模可能与成熟SKU不同，但变化趋势（涨跌比例）应该相似
6. 重点参考运营备注中的增长趋势和广告投放变化
7. 预测值必须为非负整数

请直接输出JSON格式的预测结果，不要其他解释：
{{"predictions": [{{"date": "YYYY-MM-DD", "quantity": 整数}}]}}"""


def call_llm(prompt):
    """调用 LLM via AWS Bedrock（含重试）"""
    client = boto3.client('bedrock-runtime', region_name=REGION)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    })
    for attempt in range(5):
        try:
            resp = client.invoke_model(modelId=MODEL_ID, body=body, contentType='application/json')
            return json.loads(resp['body'].read())['content'][0]['text']
        except Exception as e:
            wait = 2 ** attempt * 5
            print(f"    [重试 {attempt+1}/5] {e}, 等待{wait}s...")
            time.sleep(wait)
    raise RuntimeError("API调用失败5次")


def parse_predictions(text, expected_dates):
    """从LLM输出中解析预测值"""
    json_match = re.search(r'\{[\s\S]*"predictions"[\s\S]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            preds = {p['date']: max(0, int(p['quantity'])) for p in data['predictions']}
            return [preds.get(d.strftime('%Y-%m-%d'), 0) for d in expected_dates]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    numbers = re.findall(r'"quantity"\s*:\s*(\d+)', text)
    if len(numbers) >= len(expected_dates):
        return [int(n) for n in numbers[:len(expected_dates)]]
    print(f"  [WARN] 解析失败。LLM输出: {text[:200]}")
    return [0] * len(expected_dates)


def calc_acc(pred, actual):
    if actual > 0:
        return max(0, 1 - abs(pred - actual) / actual) * 100
    return 100.0 if pred == 0 else 0.0


def main():
    print("加载数据...")
    daily = load_data()
    cold_skus = get_cold_start_skus(daily)

    print(f"\n冷启动SKU: {len(cold_skus)}个")
    for c in cold_skus:
        print(f"  {c['sku']}: 品类={c['cat']}, 训练={c['train_nz']}天, 同品类成熟={len(c['peers'])}个")

    out_path = f'{OUT_DIR}/llm_cold_start_backtest.csv'
    results, done_skus = [], set()
    try:
        existing = pd.read_csv(out_path)
        results = existing.to_dict('records')
        done_skus = set(existing['sku'].unique())
        print(f"\n已有结果: {len(done_skus)}个SKU, 从断点继续...")
    except FileNotFoundError:
        pass

    t_total = time.time()
    for info in cold_skus:
        sku, cat = info['sku'], info['cat']
        if sku in done_skus:
            print(f"  {sku}: 已完成，跳过")
            continue

        t0 = time.time()
        test_data = daily[(daily['sku'] == sku) & (daily['date'] >= TEST_START) & (daily['date'] <= TEST_END)].sort_values('date')
        if len(test_data) == 0:
            continue

        test_dates = test_data['date'].tolist()
        all_preds, history_end = [], pd.Timestamp(TRAIN_END)

        for start in range(0, len(test_dates), ROLL_DAYS):
            end = min(start + ROLL_DAYS, len(test_dates))
            batch_dates = test_dates[start:end]
            try:
                text = call_llm(build_prompt(daily, sku, info, batch_dates, history_end))
                preds = parse_predictions(text, batch_dates)
            except Exception as e:
                print(f"  [ERROR] {sku} batch {start}: {e}")
                preds = [0] * len(batch_dates)

            all_preds.extend(preds)
            # 不推进 history_end，避免下一轮看到测试期真实数据

            actuals = test_data.iloc[start:end]['quantity'].values
            batch_accs = [calc_acc(preds[i], actuals[i]) for i in range(len(preds))]
            print(f"  {sku} [{batch_dates[0].strftime('%m/%d')}-{batch_dates[-1].strftime('%m/%d')}]: "
                  f"预测={[int(p) for p in preds]}, 实际={actuals.astype(int).tolist()}, "
                  f"准确率={np.mean(batch_accs):.1f}%")
            time.sleep(2)

        actuals = test_data['quantity'].values.astype(float)
        for i in range(len(test_data)):
            results.append({
                'sku': sku, 'category': cat, 'train_days': info['train_nz'],
                'date': test_dates[i].strftime('%Y-%m-%d'),
                'actual': int(actuals[i]), 'predicted': int(all_preds[i]),
                'accuracy': round(calc_acc(all_preds[i], actuals[i]), 1),
            })

        sku_acc = np.mean([calc_acc(all_preds[i], actuals[i]) for i in range(len(actuals))])
        print(f"{sku} (品类={cat}): 整体准确率={sku_acc:.1f}% ({time.time()-t0:.0f}s)\n")
        pd.DataFrame(results).to_csv(out_path, index=False)

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"LLM冷启动回测完成: {df['sku'].nunique()}个SKU, {len(df)}条记录")
    print(f"整体准确率: {df['accuracy'].mean():.1f}%")
    print(f">=70%占比: {(df['accuracy'] >= 70).mean()*100:.1f}%")
    print(f"\n按SKU汇总:")
    summary = df.groupby(['sku', 'category', 'train_days']).agg(
        accuracy=('accuracy', 'mean'), records=('accuracy', 'count'),
    ).reset_index().sort_values('accuracy', ascending=False)
    print(summary.to_string(index=False))
    print(f"\n总耗时: {time.time()-t_total:.0f}s | 结果: {out_path}")


if __name__ == '__main__':
    main()
