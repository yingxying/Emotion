import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="智能办公环境监测系统", layout="wide")
st.title("📊 智能办公环境与员工状态监测系统")


# ------------------------------------------------------------
# 1. 加载模型和特征名（缓存，只加载一次）
@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler.pkl")
    arousal_model = joblib.load("models/xgb_arousal_model.pkl")
    valence_model = joblib.load("models/xgb_valence_model.pkl")
    return scaler, arousal_model, valence_model


@st.cache_data
def load_data():
    zip_path = "data.zip"
    extract_to = "data/"

    # 如果 data 文件夹不存在，就解压
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    # 找到解压后的 CSV 文件并读取
    csv_path = os.path.join(extract_to, "dreamer_features.csv")  # 改成你的实际文件名
    df = pd.read_csv(csv_path)
    return df


# 加载特征列表（算法工程师给的 feature_names.txt，每行一个特征名）
with open("feature_names.txt", "r") as f:
    feature_names = [line.strip() for line in f if line.strip()]

scaler, arousal_model, valence_model = load_models()
df = load_data()

# ------------------------------------------------------------
# 2. 侧边栏：选择被试和试次
st.sidebar.header("🔍 数据筛选")
subject_list = sorted(df["subject"].unique())
trial_list = sorted(df["trial"].unique())

selected_subject = st.sidebar.selectbox("选择被试 (subject)", subject_list)
selected_trial = st.sidebar.selectbox("选择片段 (trial)", trial_list)

# 筛选数据
filtered_df = df[
    (df["subject"] == selected_subject) & (df["trial"] == selected_trial)
    ].copy()
filtered_df = filtered_df.sort_values("window_index")

if filtered_df.empty:
    st.error("未找到该被试和片段的数据，请重新选择。")
    st.stop()

st.subheader(f"📋 被试 {selected_subject} - 片段 {selected_trial} 的数据（前10行）")
st.dataframe(filtered_df.head(10))

# 方案三：自动识别特征列（不依赖 feature_names.txt）
# 定义不需要作为特征的列名（这些是标签和元数据）
non_feature_cols = ['subject', 'trial', 'window_index', 'valence', 'arousal', 'dominance']

# 自动识别特征列：除了上述列之外的所有列
feature_cols = [col for col in filtered_df.columns if col not in non_feature_cols]

# 检查是否有特征列
if len(feature_cols) == 0:
    st.error("❌ 没有找到任何特征列，请检查数据格式。")
    st.stop()

# 可选：在页面上显示找到了哪些特征列（调试用，正式运行后可删除）
with st.expander("🔍 查看自动识别的特征列（共 {} 个）".format(len(feature_cols))):
    st.write(feature_cols[:20])  # 只显示前20个，避免刷屏
    if len(feature_cols) > 20:
        st.write(f"... 还有 {len(feature_cols) - 20} 个")

# 提取特征矩阵（顺序不重要，因为模型训练时特征顺序已经固定？）
# ⚠️ 注意：XGBoost 模型不依赖特征顺序，所以可以这样用
X_raw = filtered_df[feature_cols].values

# 标准化
X_scaled = scaler.transform(X_raw)

# 模型预测
arousal_pred = arousal_model.predict(X_scaled)  # 预测的唤醒度/专注度 1-5
valence_pred = valence_model.predict(X_scaled)  # 预测的效价/心情 1-5

# 将预测结果加入 DataFrame
filtered_df["专注度预测 (1-5)"] = arousal_pred
filtered_df["心情预测 (1-5)"] = valence_pred

# 数据集中原有的 valence 和 arousal 列就是真实标签（参与者自评）
# 用于与模型预测结果进行对比
if "arousal" in filtered_df.columns and "valence" in filtered_df.columns:
    filtered_df["真实专注度"] = filtered_df["arousal"]
    filtered_df["真实心情"] = filtered_df["valence"]

# ------------------------------------------------------------
# 4. 展示预测结果表格
st.subheader("🔮 预测结果 vs 真实标签（每秒窗口）")

# 动态选择要显示的列
display_cols = ["window_index", "专注度预测 (1-5)", "心情预测 (1-5)"]
if "真实专注度" in filtered_df.columns:
    display_cols.extend(["真实专注度", "真实心情"])

st.dataframe(filtered_df[display_cols].head(20))

# 计算预测误差（如果有真实标签）
if "真实专注度" in filtered_df.columns:
    mae_arousal = np.mean(np.abs(filtered_df["专注度预测 (1-5)"] - filtered_df["真实专注度"]))
    mae_valence = np.mean(np.abs(filtered_df["心情预测 (1-5)"] - filtered_df["真实心情"]))

    col1, col2 = st.columns(2)
    col1.metric("专注度预测平均绝对误差 (MAE)", f"{mae_arousal:.3f}")
    col2.metric("心情预测平均绝对误差 (MAE)", f"{mae_valence:.3f}")

# ------------------------------------------------------------
# 5. 可视化趋势（用 window_index 作为 x 轴）
st.subheader("📈 专注度趋势（每秒）")
st.line_chart(filtered_df.set_index("window_index")["专注度预测 (1-5)"])

st.subheader("😊 心情趋势（每秒）")
st.line_chart(filtered_df.set_index("window_index")["心情预测 (1-5)"])

# 如果存在真实标签，画对比图
if "真实专注度" in filtered_df.columns:
    st.subheader("🔄 预测 vs 真实（专注度）")
    compare_arousal = filtered_df[["window_index", "专注度预测 (1-5)", "真实专注度"]].set_index("window_index")
    st.line_chart(compare_arousal)

    st.subheader("🔄 预测 vs 真实（心情）")
    compare_valence = filtered_df[["window_index", "心情预测 (1-5)", "真实心情"]].set_index("window_index")
    st.line_chart(compare_valence)

# ------------------------------------------------------------
# 6. 需要关注的时段（专注度<2 或 心情<2）
st.subheader("⚠️ 需要关注的时段")
alert_df = filtered_df[
    (filtered_df["专注度预测 (1-5)"] < 2) | (filtered_df["心情预测 (1-5)"] < 2)
    ].copy()
if not alert_df.empty:
    st.dataframe(alert_df[["window_index", "专注度预测 (1-5)", "心情预测 (1-5)"]])
    st.warning(f"⚠️ 共有 {len(alert_df)} 秒处于低唤醒或低愉悦状态，建议关注。")
else:
    st.success("✅ 当前片段暂无异常，员工状态良好。")

# ------------------------------------------------------------
# 7. 整体统计卡片
col1, col2, col3 = st.columns(3)
col1.metric("平均专注度", f"{filtered_df['专注度预测 (1-5)'].mean():.2f}")
col2.metric("平均心情", f"{filtered_df['心情预测 (1-5)'].mean():.2f}")

# 高效工作占比：专注度>3 且 心情>3 的窗口比例
effective_ratio = ((filtered_df["专注度预测 (1-5)"] > 3) &
                   (filtered_df["心情预测 (1-5)"] > 3)).mean()
col3.metric("高效工作占比", f"{effective_ratio:.1%}")

st.caption(f"数据更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("💡 说明：真实专注度/心情来自数据集中参与者自评（arousal/valence列）")