"""Streamlit 主应用"""
import streamlit as st

st.set_page_config(
    page_title="Amazon 销量预测系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🛒 Amazon 销量预测系统")
    st.markdown("---")
    
    st.markdown("""
    ### 欢迎使用 Amazon 销量预测系统
    
    本系统采用 **Chronos + LightGBM 混合模型**，结合促销效果校准，帮助您准确预测未来销量。
    
    #### 🎯 核心特性
    - ✅ 促销旺季预测准确率 **80%+**
    - ✅ 支持输入 **未来促销计划**（折扣率、广告预算）
    - ✅ 多情景预测和置信区间
    - ✅ 数据完全本地处理，**不上传互联网**
    
    #### 📋 使用流程
    1. **数据上传** - 上传历史销售数据
    2. **模型训练** - 系统自动训练预测模型
    3. **计划输入** - 输入未来的促销和广告计划
    4. **销量预测** - 查看预测结果和可视化
    
    👈 请在左侧边栏选择功能页面开始使用
    """)
    
    st.markdown("---")
    st.markdown("### 📈 当前状态")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.get('data_uploaded'):
            st.success("✅ 数据已上传")
            st.metric("数据天数", st.session_state.get('data_days', 0))
        else:
            st.warning("⏳ 待上传数据")
    
    with col2:
        if st.session_state.get('model_trained'):
            st.success("✅ 模型已训练")
            st.metric("回测准确率", f"{st.session_state.get('accuracy', 0):.1%}")
        else:
            st.warning("⏳ 待训练模型")
    
    with col3:
        if st.session_state.get('forecast_generated'):
            st.success("✅ 预测已生成")
            st.metric("预测天数", st.session_state.get('forecast_days', 0))
        else:
            st.info("⏳ 待生成预测")

if __name__ == "__main__":
    main()
