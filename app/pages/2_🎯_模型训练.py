"""页面2：模型训练"""
import streamlit as st
import pandas as pd
import sys
sys.path.insert(0, '/home/ec2-user/sales_forecast_system')
from src.features.engineer import FeatureEngineer
from src.models.ensemble import EnsembleForecaster
from src.calibration.calibrator import PromotionCalibrator

st.title("🎯 模型训练")

if 'processed_data' not in st.session_state:
    st.warning("⚠️ 请先上传数据")
    st.stop()

df = st.session_state.processed_data
stats = st.session_state.get('historical_stats', {})

st.markdown(f"""
### 当前数据
- 数据天数：**{len(df)}** 天
- 日期范围：**{df['date'].min().strftime('%Y-%m-%d')}** ~ **{df['date'].max().strftime('%Y-%m-%d')}**
- 日均销量：**{stats.get('avg_daily_sales', 0):.0f}**
""")

st.markdown("---")
st.markdown("### ⚙️ 训练配置")

col1, col2 = st.columns(2)
with col1:
    use_chronos = st.checkbox("使用 Chronos 模型", value=True, help="深度学习时序模型，更准确但更慢")
    val_days = st.slider("验证集天数", 14, 90, 30)
with col2:
    use_calibration = st.checkbox("启用促销校准", value=True, help="基于历史促销效果校准预测")

if st.button("🚀 开始训练", type="primary"):
    progress = st.progress(0)
    status = st.empty()
    
    try:
        # Step 1: 特征工程
        status.text("Step 1/3: 特征工程...")
        progress.progress(20)
        
        fe = FeatureEngineer(stats)
        df_features = fe.create_features(df)
        
        # Step 2: 训练模型
        status.text("Step 2/3: 训练模型...")
        progress.progress(50)
        
        model = EnsembleForecaster(use_chronos=use_chronos)
        result = model.fit(df_features, val_days=val_days)
        
        # Step 3: 学习促销效果
        status.text("Step 3/3: 分析促销效果...")
        progress.progress(80)
        
        calibrator = PromotionCalibrator.from_data(df_features) if use_calibration else None
        
        progress.progress(100)
        status.text("✅ 训练完成！")
        
        # 保存
        st.session_state.model = model
        st.session_state.feature_engineer = fe
        st.session_state.calibrator = calibrator
        st.session_state.model_trained = True
        st.session_state.accuracy = result['accuracy']
        st.session_state.training_result = result
        
        st.success("🎉 模型训练完成！")
        
        # 显示结果
        st.markdown("---")
        st.markdown("### 📊 训练报告")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("回测准确率 (±20%)", f"{result['accuracy']:.1%}")
        with col2:
            st.metric("MAPE", f"{result['mape']:.1%}")
        with col3:
            w = result['weights']
            st.metric("模型权重", f"C:{w['chronos']:.0%}/L:{w['lgb']:.0%}/X:{w['xgb']:.0%}")
        
        # 特征重要性
        st.markdown("### 🔍 特征重要性 Top 10")
        imp = result['feature_importance'].head(10)
        st.bar_chart(imp.set_index('feature')['importance'])
        
        st.info("👉 请前往 **计划输入** 页面，输入未来促销计划")
        
    except Exception as e:
        st.error(f"❌ 训练失败: {str(e)}")
        raise e
