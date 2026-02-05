"""页面3：未来计划输入"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.title("📅 未来计划输入")

if not st.session_state.get('model_trained'):
    st.warning("⚠️ 请先训练模型")
    st.stop()

st.markdown("""
### 输入未来促销和广告计划

这是提高预测准确率的关键步骤！请输入您计划的：
- 📌 **促销日期和类型**
- 💰 **折扣率**
- 📢 **广告预算**
""")

st.markdown("---")
st.markdown("### 📆 预测日期范围")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("开始日期", value=datetime.now().date() + timedelta(days=1))
with col2:
    end_date = st.date_input("结束日期", value=datetime.now().date() + timedelta(days=30))

pred_days = (end_date - start_date).days + 1
st.info(f"📊 将预测 **{pred_days}** 天的销量")

st.markdown("---")
st.markdown("### ✏️ 促销计划设置")

if 'future_plan' not in st.session_state:
    st.session_state.future_plan = []

# 添加促销
st.markdown("**添加促销活动：**")
c1, c2, c3, c4 = st.columns(4)

with c1:
    promo_start = st.date_input("促销开始", value=start_date, key="ps")
with c2:
    promo_end = st.date_input("促销结束", value=promo_start, key="pe")
with c3:
    promo_type = st.selectbox("促销类型", 
        ['prime_day', 'black_friday', 'cyber_monday', 'lightning_deal', 'coupon', 'other'],
        format_func=lambda x: {'prime_day': '🎯 Prime Day', 'black_friday': '🛍️ Black Friday',
            'cyber_monday': '💻 Cyber Monday', 'lightning_deal': '⚡ Lightning Deal',
            'coupon': '🎫 Coupon', 'other': '📌 其他'}.get(x, x))
with c4:
    promo_discount = st.slider("折扣率 (%)", 0, 50, 20)

c5, c6 = st.columns(2)
with c5:
    promo_ppc = st.number_input("每日广告预算 ($)", 0, 10000, 100, step=10)
with c6:
    avg_ppc = st.session_state.get('historical_stats', {}).get('avg_ppc_fee', 50)
    ppc_ratio = promo_ppc / avg_ppc if avg_ppc > 0 else 1
    st.metric("广告预算倍数", f"{ppc_ratio:.1f}x")

if st.button("➕ 添加促销"):
    dates = pd.date_range(promo_start, promo_end)
    existing = [p['date'] for p in st.session_state.future_plan]
    for d in dates:
        ds = str(d.date())
        if ds not in existing:
            st.session_state.future_plan.append({
                'date': ds, 'is_promotion': True, 'promotion_type': promo_type,
                'discount_rate': promo_discount / 100, 'ppc_budget': promo_ppc,
                'ppc_budget_ratio': ppc_ratio
            })
    st.success(f"✅ 已添加 {len(dates)} 天促销计划")
    st.rerun()

# 非促销日默认
st.markdown("---")
st.markdown("**非促销日设置：**")
dc1, dc2 = st.columns(2)
with dc1:
    default_discount = st.slider("默认折扣率 (%)", 0, 20, 0)
with dc2:
    default_ppc = st.number_input("默认广告预算 ($)", 0, 1000, 50, step=10)

# 显示当前计划
st.markdown("---")
st.markdown("### 📋 当前计划预览")

if st.session_state.future_plan:
    plan_df = pd.DataFrame(st.session_state.future_plan)
    promo_days = plan_df[plan_df['is_promotion'] == True]
    
    if not promo_days.empty:
        st.dataframe(promo_days, use_container_width=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("促销天数", len(promo_days))
    with c2:
        avg_d = promo_days['discount_rate'].mean() * 100 if not promo_days.empty else 0
        st.metric("平均折扣率", f"{avg_d:.0f}%")
    with c3:
        total_ppc = promo_days['ppc_budget'].sum() if not promo_days.empty else 0
        st.metric("促销期广告总预算", f"${total_ppc:,.0f}")
    
    if st.button("🗑️ 清除所有计划"):
        st.session_state.future_plan = []
        st.rerun()
else:
    st.info("暂无促销计划")

# 确认
st.markdown("---")
if st.button("✅ 确认计划并继续预测", type="primary"):
    all_dates = pd.date_range(start_date, end_date)
    existing = [p['date'] for p in st.session_state.future_plan]
    
    complete_plan = st.session_state.future_plan.copy()
    for d in all_dates:
        ds = str(d.date())
        if ds not in existing:
            complete_plan.append({
                'date': ds, 'is_promotion': False, 'promotion_type': 'none',
                'discount_rate': default_discount / 100, 'ppc_budget': default_ppc,
            })
    
    complete_plan.sort(key=lambda x: x['date'])
    
    st.session_state.complete_plan = complete_plan
    st.session_state.prediction_start = start_date
    st.session_state.prediction_end = end_date
    st.session_state.plan_confirmed = True
    
    st.success("✅ 计划已确认！")
    st.info("👉 请前往 **销量预测** 页面查看预测结果")
