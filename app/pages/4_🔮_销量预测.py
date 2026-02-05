"""页面4：销量预测"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
import sys
sys.path.insert(0, '/home/ec2-user/sales_forecast_system')
from src.features.engineer import FeatureEngineer

st.title("🔮 销量预测")

if not st.session_state.get('model_trained'):
    st.warning("⚠️ 请先训练模型")
    st.stop()

if not st.session_state.get('plan_confirmed'):
    st.warning("⚠️ 请先输入未来计划")
    st.stop()

model = st.session_state.model
fe = st.session_state.feature_engineer
calibrator = st.session_state.get('calibrator')
complete_plan = st.session_state.complete_plan

promo_count = sum(1 for p in complete_plan if p.get('is_promotion'))
st.markdown(f"""
### 预测配置
- 📆 预测范围：**{st.session_state.prediction_start}** ~ **{st.session_state.prediction_end}**
- 📊 预测天数：**{len(complete_plan)}** 天
- 🎯 促销天数：**{promo_count}** 天
""")

if st.button("🚀 生成预测", type="primary"):
    with st.spinner("生成预测中..."):
        try:
            # 构建预测数据
            pred_dates = pd.date_range(st.session_state.prediction_start, 
                                       st.session_state.prediction_end)
            pred_df = pd.DataFrame({'date': pred_dates})
            
            # 添加特征
            pred_df = fe.create_features(pred_df, for_prediction=True)
            pred_df = fe.add_plan_features(pred_df, complete_plan)
            
            # 预测
            forecast = model.predict(pred_df, len(pred_dates))
            
            # 校准
            if calibrator:
                forecast = calibrator.calibrate(forecast, complete_plan)
            
            # 添加计划信息
            plan_df = pd.DataFrame(complete_plan)
            plan_df['date'] = pd.to_datetime(plan_df['date'])
            forecast = forecast.merge(plan_df, on='date', how='left')
            
            st.session_state.forecast_result = forecast
            st.session_state.forecast_generated = True
            st.session_state.forecast_days = len(complete_plan)
            
            st.success("✅ 预测完成！")
        except Exception as e:
            st.error(f"❌ 预测失败: {str(e)}")
            raise e

# 显示结果
if st.session_state.get('forecast_result') is not None:
    forecast = st.session_state.forecast_result
    
    st.markdown("---")
    st.markdown("### 📈 预测结果")
    
    # 汇总
    total_pred = forecast['prediction'].sum()
    avg_pred = forecast['prediction'].mean()
    promo_mask = forecast.get('is_promotion', pd.Series([False]*len(forecast))) == True
    promo_sales = forecast.loc[promo_mask, 'prediction'].sum() if promo_mask.any() else 0
    promo_ratio = promo_sales / total_pred if total_pred > 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("预测总销量", f"{total_pred:,.0f}")
    with c2:
        st.metric("日均销量", f"{avg_pred:,.0f}")
    with c3:
        st.metric("促销期销量", f"{promo_sales:,.0f}")
    with c4:
        st.metric("促销期占比", f"{promo_ratio:.1%}")
    
    # 图表
    st.markdown("### 📊 预测趋势图")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['date'], y=forecast['prediction'],
        mode='lines+markers', name='预测销量', line=dict(color='#1f77b4', width=2)))
    
    fig.add_trace(go.Scatter(
        x=list(forecast['date']) + list(forecast['date'])[::-1],
        y=list(forecast['upper_bound']) + list(forecast['lower_bound'])[::-1],
        fill='toself', fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'), name='80% 置信区间'))
    
    if promo_mask.any():
        promo_data = forecast[promo_mask]
        fig.add_trace(go.Scatter(x=promo_data['date'], y=promo_data['prediction'],
            mode='markers', name='促销日', marker=dict(size=12, color='red', symbol='star')))
    
    fig.update_layout(title='销量预测趋势', xaxis_title='日期', yaxis_title='销量',
                      hovermode='x unified', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # 详细数据
    st.markdown("### 📋 详细预测数据")
    
    display_cols = ['date', 'prediction', 'lower_bound', 'upper_bound']
    if 'is_promotion' in forecast.columns:
        display_cols.extend(['is_promotion', 'promotion_type', 'discount_rate', 'ppc_budget'])
    
    display_df = forecast[[c for c in display_cols if c in forecast.columns]].copy()
    display_df['prediction'] = display_df['prediction'].round(0).astype(int)
    display_df['lower_bound'] = display_df['lower_bound'].round(0).astype(int)
    display_df['upper_bound'] = display_df['upper_bound'].round(0).astype(int)
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # 多情景
    st.markdown("### 🎭 多情景分析")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**😔 悲观情景**")
        st.metric("预测总销量", f"{forecast['lower_bound'].sum():,.0f}")
    with c2:
        st.markdown("**😐 基准情景**")
        st.metric("预测总销量", f"{total_pred:,.0f}")
    with c3:
        st.markdown("**😊 乐观情景**")
        st.metric("预测总销量", f"{forecast['upper_bound'].sum():,.0f}")
    
    # 导出
    st.markdown("---")
    st.markdown("### 💾 导出预测结果")
    
    c1, c2 = st.columns(2)
    with c1:
        csv = forecast.to_csv(index=False).encode('utf-8')
        st.download_button("📥 下载 CSV", csv, 
            f"forecast_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    with c2:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            forecast.to_excel(writer, sheet_name='预测结果', index=False)
        st.download_button("📥 下载 Excel", buffer.getvalue(),
            f"forecast_{datetime.now().strftime('%Y%m%d')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
