#!/usr/bin/env python3
"""采集外部数据：Google Trends + 美国节假日日历"""
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta

OUT_DIR = '/home/ec2-user/sales_forecast_system/data/external'
os.makedirs(OUT_DIR, exist_ok=True)

# ========== 1. Google Trends ==========
def fetch_google_trends():
    from pytrends.request import TrendReq
    
    keywords_by_category = {
        'LED Strip Lights': ['LED strip lights', 'RGBIC LED lights', 'smart LED strip'],
        'Hygrometer': ['hygrometer', 'indoor thermometer', 'WiFi thermometer'],
        'Permanent Outdoor Lights': ['permanent outdoor lights', 'eave lights', 'outdoor LED lights'],
        'Smart Plug': ['smart plug', 'WiFi smart plug', 'smart outlet'],
        'Smart Light Bulb': ['smart light bulb', 'color changing light bulb'],
        'TV Backlight': ['TV backlight', 'TV LED backlight', 'bias lighting'],
        'Car LED': ['car LED lights', 'interior car lights', 'underglow lights'],
        'Meat Thermometer': ['meat thermometer', 'bluetooth meat thermometer'],
        'Water Leak Detector': ['water leak detector', 'water alarm sensor'],
        'Brand': ['Govee', 'Govee lights'],
    }
    
    pytrends = TrendReq(hl='en-US', tz=300)  # US Eastern
    all_data = []
    
    # Google Trends 一次最多5个关键词，按批次查询
    # 时间范围：2023-01-01 ~ 2025-11-29
    timeframe = '2023-01-01 2025-11-29'
    
    for category, kws in keywords_by_category.items():
        for kw in kws:
            print(f'  Google Trends: {kw}...', flush=True)
            try:
                pytrends.build_payload([kw], cat=0, timeframe=timeframe, geo='US')
                data = pytrends.interest_over_time()
                if not data.empty:
                    for date, row in data.iterrows():
                        all_data.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'keyword': kw,
                            'category': category,
                            'google_trends_index': int(row[kw]),
                            'region': 'US',
                        })
                time.sleep(2)  # 避免被限流
            except Exception as e:
                print(f'    失败: {e}', flush=True)
                time.sleep(5)
    
    if all_data:
        df = pd.DataFrame(all_data)
        path = f'{OUT_DIR}/search_trends.csv'
        df.to_csv(path, index=False)
        print(f'  保存: {path} ({len(df)}条)', flush=True)
        return df
    return None


# ========== 2. 美国节假日+购物节日历 ==========
def generate_holiday_calendar():
    """生成2023-2025年美国节假日和购物节日历"""
    holidays = []
    
    for year in [2023, 2024, 2025]:
        events = [
            (f'{year}-01-01', "New Year's Day"),
            (f'{year}-01-02', "New Year's Day (observed)" if year == 2023 else None),
            (f'{year}-02-14', "Valentine's Day"),
            (f'{year}-05-14' if year == 2023 else f'{year}-05-12' if year == 2024 else f'{year}-05-11', "Mother's Day"),
            (f'{year}-06-18' if year == 2023 else f'{year}-06-16' if year == 2024 else f'{year}-06-15', "Father's Day"),
            (f'{year}-07-04', "Independence Day"),
            # Prime Day (通常7月中旬)
            (f'{year}-07-11', "Prime Day (Day 1)"),
            (f'{year}-07-12', "Prime Day (Day 2)"),
            # 返校季
            (f'{year}-08-01', "Back to School Season Start"),
            # Labor Day (9月第1个周一)
            (f'{year}-09-04' if year == 2023 else f'{year}-09-02' if year == 2024 else f'{year}-09-01', "Labor Day"),
            # Prime Big Deal Days (10月)
            (f'{year}-10-10', "Prime Big Deal Days (Day 1)"),
            (f'{year}-10-11', "Prime Big Deal Days (Day 2)"),
            # Halloween
            (f'{year}-10-31', "Halloween"),
            # 感恩节 (11月第4个周四)
            (f'{year}-11-23' if year == 2023 else f'{year}-11-28' if year == 2024 else f'{year}-11-27', "Thanksgiving"),
            # 黑五 (感恩节后一天)
            (f'{year}-11-24' if year == 2023 else f'{year}-11-29' if year == 2024 else f'{year}-11-28', "Black Friday"),
            # 网一 (黑五后的周一)
            (f'{year}-11-27' if year == 2023 else f'{year}-12-02' if year == 2024 else f'{year}-12-01', "Cyber Monday"),
            # 圣诞
            (f'{year}-12-25', "Christmas Day"),
        ]
        for date_str, name in events:
            if name:
                holidays.append({'date': date_str, 'holiday_name': name, 'is_holiday': 1})
    
    # 生成完整日期范围
    date_range = pd.date_range('2023-01-01', '2025-11-29')
    holiday_df = pd.DataFrame(holidays)
    holiday_df['date'] = pd.to_datetime(holiday_df['date'])
    
    full_df = pd.DataFrame({'date': date_range})
    full_df = full_df.merge(holiday_df, on='date', how='left')
    full_df['is_holiday'] = full_df['is_holiday'].fillna(0).astype(int)
    full_df['holiday_name'] = full_df['holiday_name'].fillna('')
    
    # 添加购物季标记
    full_df['shopping_season'] = ''
    for _, row in full_df.iterrows():
        m, d = row['date'].month, row['date'].day
        if m == 7 and 10 <= d <= 13:
            full_df.loc[full_df['date'] == row['date'], 'shopping_season'] = 'Prime Day'
        elif m == 8:
            full_df.loc[full_df['date'] == row['date'], 'shopping_season'] = 'Back to School'
        elif m == 10 and 9 <= d <= 12:
            full_df.loc[full_df['date'] == row['date'], 'shopping_season'] = 'Prime Big Deal Days'
        elif m == 11 and d >= 20:
            full_df.loc[full_df['date'] == row['date'], 'shopping_season'] = 'Black Friday Week'
        elif m == 12 and d <= 25:
            full_df.loc[full_df['date'] == row['date'], 'shopping_season'] = 'Holiday Season'
    
    full_df['date'] = full_df['date'].dt.strftime('%Y-%m-%d')
    path = f'{OUT_DIR}/us_holidays.csv'
    full_df.to_csv(path, index=False)
    print(f'  保存: {path} ({len(full_df)}条)', flush=True)
    return full_df


# ========== 3. FRED 宏观经济数据 ==========
def fetch_fred_data():
    """尝试从FRED获取消费者信心和零售数据"""
    try:
        from fredapi import Fred
        # FRED API key (免费注册获取: https://fred.stlouisfed.org/docs/api/api_key.html)
        api_key = os.environ.get('FRED_API_KEY', '')
        if not api_key:
            print('  FRED_API_KEY 未设置，跳过（可在 https://fred.stlouisfed.org 免费注册获取）', flush=True)
            return None
        
        fred = Fred(api_key=api_key)
        
        series = {
            'UMCSENT': 'consumer_sentiment',      # 密歇根消费者信心指数
            'RSXFS': 'retail_sales',               # 零售销售（除食品服务）
            'PCEPILFE': 'core_pce',                # 核心PCE通胀
        }
        
        all_data = []
        for series_id, name in series.items():
            print(f'  FRED: {series_id} ({name})...', flush=True)
            try:
                data = fred.get_series(series_id, observation_start='2023-01-01', observation_end='2025-11-29')
                for date, value in data.items():
                    all_data.append({'date': date.strftime('%Y-%m-%d'), 'indicator': name, 'value': value})
            except Exception as e:
                print(f'    失败: {e}', flush=True)
        
        if all_data:
            df = pd.DataFrame(all_data)
            path = f'{OUT_DIR}/macro_data.csv'
            df.to_csv(path, index=False)
            print(f'  保存: {path} ({len(df)}条)', flush=True)
            return df
    except ImportError:
        print('  fredapi 未安装', flush=True)
    return None


def main():
    print('=== 外部数据采集 ===\n', flush=True)
    
    print('[1/3] 美国节假日日历...', flush=True)
    generate_holiday_calendar()
    
    print('\n[2/3] Google Trends...', flush=True)
    fetch_google_trends()
    
    print('\n[3/3] FRED 宏观数据...', flush=True)
    fetch_fred_data()
    
    print('\n=== 完成 ===', flush=True)
    print(f'数据保存在: {OUT_DIR}/', flush=True)
    print('\n待补充（需要API key或手动采集）:', flush=True)
    print('  - Keepa API → 价格历史 + BSR数据（需购买API key）', flush=True)
    print('  - Jungle Scout / Helium 10 → 品类大盘数据（需账号）', flush=True)
    print('  - Amazon 竞品页面 → 竞品价格/评分/库存（需爬虫）', flush=True)


if __name__ == '__main__':
    main()
