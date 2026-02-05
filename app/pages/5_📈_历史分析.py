"""页面5：历史分析"""
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 历史数据分析")

if 'processed_data' not in st.session_state:
    st.warning("⚠️ 请先上传数据")
    st.stop()

df = st.session_state.processed_data

# 时间范围
st.markdown("### 📆 分析范围")
c1, c2 = st.columns(2)
with c1:
    start = st.date_input("开始日期", value=df['date'].min())
with c2:
    end = st.date_input("结束日期", value=df['date'].max())

mask = (df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))
filtered = df[mask]

# 概览
st.markdown("---")
st.markdown("### 📊 数据概览")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("总天数", len(filtered))
with c2:
    st.metric("总销量", f"{filtered['quantity'].sum():,.0f}")
with c3:
    st.metric("日均销量", f"{filtered['quantity'].mean():.0f}")
with c4:
    st.metric("销量标准差", f"{filtered['quantity'].std():.0f}")

# 趋势
st.markdown("---")
st.markdown("### 📈 销量趋势")
fig = px.line(filtered, x='date', y='quantity', title='日销量趋势')
st.plotly_chart(fig, use_container_width=True)

# 周度
st.markdown("### 📅 周度分析")
filtered['dow'] = filtered['date'].dt.dayofweek
weekday = filtered.groupby('dow')['quantity'].mean().reset_index()
weekday['day'] = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
fig = px.bar(weekday, x='day', y='quantity', title='各星期销量分布')
st.plotly_chart(fig, use_container_width=True)

# 月度
st.markdown("### 📆 月度分析")
filtered['month'] = filtered['date'].dt.to_period('M').astype(str)
monthly = filtered.groupby('month')['quantity'].sum().reset_index()
fig = px.bar(monthly, x='month', y='quantity', title='月度销量')
st.plotly_chart(fig, use_container_width=True)

# 折扣分析
if 'discount_rate' in filtered.columns:
    st.markdown("---")
    st.markdown("### 💰 折扣与销量关系")
    fig = px.scatter(filtered, x='discount_rate', y='quantity', trendline='lowess',
                     title='折扣率 vs 销量')
    st.plotly_chart(fig, use_container_width=True)

# 广告分析
if 'ppc_fee' in filtered.columns:
    st.markdown("### 📢 广告投入与销量关系")
    fig = px.scatter(filtered, x='ppc_fee', y='quantity', trendline='lowess',
                     title='广告费用 vs 销量')
    st.plotly_chart(fig, use_container_width=True)
