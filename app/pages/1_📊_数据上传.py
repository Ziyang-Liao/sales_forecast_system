"""页面1：数据上传"""
import streamlit as st
import pandas as pd
import sys
sys.path.insert(0, '/home/ec2-user/sales_forecast_system')
from src.data.processor import DataProcessor

st.title("📊 数据上传")

st.markdown("""
### 上传历史销售数据

支持格式：CSV, Excel (.xlsx)

**必须字段：** `purchase_date`, `quantity`

**推荐字段：** `discount_rate`, `ppc_fee`, `sessions`, `sku`
""")

uploaded_file = st.file_uploader("选择数据文件", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        processor = DataProcessor()
        
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ 文件读取成功！共 {len(raw_df)} 条记录")
        st.dataframe(raw_df.head(10), use_container_width=True)
        
        if st.button("🔍 验证并处理数据", type="primary"):
            with st.spinner("处理中..."):
                df = processor.load_data(raw_df)
                is_valid, errors = processor.validate(df)
                
                if is_valid:
                    # 按日聚合
                    daily_df = processor.aggregate_daily(df)
                    stats = processor.compute_stats(daily_df)
                    
                    st.session_state.raw_data = raw_df
                    st.session_state.processed_data = daily_df
                    st.session_state.historical_stats = stats
                    st.session_state.data_uploaded = True
                    st.session_state.data_days = len(daily_df)
                    
                    st.success("✅ 数据验证通过！")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("总天数", len(daily_df))
                    with col2:
                        st.metric("日均销量", f"{stats['avg_daily_sales']:.0f}")
                    with col3:
                        st.metric("总销量", f"{daily_df['quantity'].sum():,.0f}")
                    with col4:
                        date_range = f"{daily_df['date'].min().strftime('%Y-%m-%d')} ~ {daily_df['date'].max().strftime('%Y-%m-%d')}"
                        st.metric("日期范围", date_range)
                    
                    st.info("👉 请前往 **模型训练** 页面继续")
                else:
                    st.error("❌ 数据验证失败")
                    for e in errors:
                        st.warning(e)
    except Exception as e:
        st.error(f"❌ 处理失败: {str(e)}")

# 示例数据
st.markdown("---")
st.markdown("### 📥 示例数据")

@st.cache_data
def gen_sample():
    import numpy as np
    np.random.seed(42)
    dates = pd.date_range('2023-06-01', '2024-11-15')
    n = len(dates)
    return pd.DataFrame({
        'purchase_date': [d.strftime('%Y%m%d') for d in dates],
        'quantity': [max(1, int(100 * (1 + 0.1*np.sin(i/30) + 0.2*np.random.randn()))) for i in range(n)],
        'discount_rate': [0.05 if i % 30 > 25 else 0 for i in range(n)],
        'ppc_fee': [50 + 20*np.random.random() for _ in range(n)],
    })

csv = gen_sample().to_csv(index=False).encode('utf-8')
st.download_button("📥 下载示例数据", csv, "sample_data.csv", "text/csv")
