"""
三层因果可解释诊断系统 — Streamlit Web 应用
让教师通过浏览器即可使用计算思维诊断工具
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CAUSAL_ORDER, DIM_NAMES, FEATURE_COLUMNS
from src.data import prepare_simulated_data, simulate_intervention_data
from src.model import train_model, save_model, load_model
from src.layer1_cf import layer1_counterfactual
from src.layer2_diag import layer2_dual_diagnosis, compute_adaptive_thresholds
from src.layer3_ite import layer3_estimate_ite
from src.visualize import plot_dual_diagnosis
from src.llm import generate_teacher_report

# ── 页面配置 ──────────────────────────────────────────────────

st.set_page_config(
    page_title="计算思维诊断系统",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 缓存：模型只训练一次 ──────────────────────────────────────

@st.cache_resource
def get_or_train_model(_df):
    """缓存模型，避免每次交互都重新训练"""
    model, X_train, X_test, rmse = train_model(_df)
    shap_threshold, ce_threshold = compute_adaptive_thresholds(model, X_train)
    return model, X_train, X_test, rmse, shap_threshold, ce_threshold


# ── 标题 ──────────────────────────────────────────────────────

st.title("🎓 三层因果可解释诊断系统")
st.caption(
    "基于 XGBoost + SHAP + 因果推断（后门调整），"
    "为计算思维教学提供数据驱动的诊断与干预建议"
)

# ── 侧边栏 ────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 配置")

    # 数据模式
    st.subheader("📂 数据来源")
    data_mode = st.radio(
        "选择数据模式",
        ["📊 模拟演示数据", "📁 上传真实数据"],
        help="模拟数据用于快速体验系统功能；上传真实数据用于实际教学诊断"
    )

    if data_mode == "📁 上传真实数据":

        # ── CSV 格式说明（可折叠） ──
        with st.expander("📋 CSV文件格式说明（必读）", expanded=True):
            st.markdown("""
            **CSV 文件需包含以下列（列名需完全一致）：**

            | 列名 | 说明 | 取值范围 |
            |------|------|----------|
            | `A1` ~ `A15` | 15道题目的得分（每题1-5分） | 1 ~ 5 整数 |
            | `作业质量` | 学生的综合编程作业成绩 | 20 ~ 100 分 |

            **可选列（若已预先计算好维度均值）：**

            | 列名 | 对应题目 |
            |------|----------|
            | `抽象` | A4, A5, A6 的均值 |
            | `分解` | A1, A2, A3 的均值 |
            | `算法设计` | A9, A10 的均值 |
            | `建模` | A7, A8 的均值 |
            | `评估` | A11, A12, A13, A14, A15 的均值 |

            > 💡 如果没有预先计算维度列，系统会自动根据 A1-A15 的题目得分计算。
            """)

            # 下载模板
            template_df = pd.DataFrame({
                **{f'A{i}': [3, 4] for i in range(1, 16)},
                '作业质量': [65.0, 78.0],
            })
            csv_data = template_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="⬇️ 下载CSV模板（含示例数据）",
                data=csv_data,
                file_name="学生计算思维数据_模板.csv",
                mime="text/csv",
            )

        uploaded_file = st.file_uploader(
            "上传CSV文件",
            type=["csv"],
            help="CSV需包含A1-A15指标列和「作业质量」列。编码建议UTF-8。"
        )
        if uploaded_file is not None:
            try:
                from src.data import prepare_from_csv
                df = prepare_from_csv(uploaded_file)
                st.success(f"✅ 已加载 {len(df)} 条学生数据")
            except Exception as e:
                st.error(f"❌ 数据加载失败：{e}")
                st.stop()
        else:
            st.info("👆 请上传CSV文件，或切换到模拟数据模式体验")
            st.stop()
    else:
        n_students = st.slider("生成学生数量", 50, 500, 200, 50)
        if "df" not in st.session_state or st.button("🔄 重新生成数据"):
            st.session_state.df = prepare_simulated_data(n=n_students)
        df = st.session_state.df
        st.success(f"✅ 已生成 {len(df)} 条模拟数据")

    # 模型
    st.subheader("🤖 模型")
    model_path = "models/xgb_model.joblib"

    if st.button("🔄 重新训练模型"):
        st.cache_resource.clear()
        st.rerun()

    with st.spinner("训练模型中..."):
        model, X_train, X_test, rmse, shap_threshold, ce_threshold = \
            get_or_train_model(df)

    st.metric("模型 RMSE", f"{rmse:.1f}分")
    st.metric("SHAP 阈值", f"{shap_threshold:.3f}")
    st.metric("因果效应阈值", f"{ce_threshold:.1f}分")

    # 模型保存
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 保存模型"):
            save_model(model, model_path)
            st.success("模型已保存")
    with col2:
        if st.button("📂 加载模型"):
            try:
                model = load_model(model_path)
                st.success("模型已加载")
            except FileNotFoundError:
                st.warning("未找到已保存的模型")

    st.divider()

    # 学生选择
    st.subheader("👤 学生选择")
    student_indices = list(X_test.index)
    selected_idx = st.selectbox(
        "选择要诊断的学生",
        student_indices,
        format_func=lambda i: f"学生 S{i:03d} (预测: {model.predict(df.loc[i:i, CAUSAL_ORDER])[0]:.0f}分)"
    )
    student_id = f"S{selected_idx:03d}"
    sample_student = X_test.loc[selected_idx]


# ── 主区域：三层分析 ──────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Layer 1 · 反事实分析",
    "🎯 Layer 2 · 双维度诊断",
    "📊 Layer 3 · 干预效应",
    "📝 教师诊断报告",
])

# ── Tab 1: Layer 1 ───────────────────────────────────────────

with tab1:
    st.subheader("Layer 1: 反事实分析（CFE）")
    st.caption(
        "⚠️ 免责声明：以下结果为模型层面的假设模拟，仅展示"
        "「若该维度独立变化，模型预测如何改变」。实际干预效果请以 Layer 2 为准。"
    )

    cf_report, cf_best_dim, cf_best_gain, all_dfs = \
        layer1_counterfactual(model, sample_student, student_id)

    base_pred = model.predict(sample_student.values.reshape(1, -1))[0]
    st.metric("当前预测作业质量", f"{base_pred:.1f}分/100分")

    st.divider()
    st.write("**各维度提升潜力模拟（固定其他维度不变）**")

    cols = st.columns(len(CAUSAL_ORDER))
    for i, feat in enumerate(CAUSAL_ORDER):
        with cols[i]:
            st.write(f"**{DIM_NAMES[feat]}**")
            df_cf = all_dfs[feat]
            st.dataframe(df_cf, hide_index=True, use_container_width=True)

    st.divider()
    st.info(f"""
    **反事实结论（模型模拟，非真实因果）**
    - 优先提升维度：**{DIM_NAMES[cf_best_dim]}**
    - 若从当前 {sample_student[cf_best_dim]:.1f} 分提升至满分
    - 模型预测变化：{base_pred:.1f} → {base_pred + cf_best_gain:.1f} (+{cf_best_gain:.1f}分)
    - ⚠️ 以上为模型假设，真实效果请以 Layer 2 因果效应为准
    """)

# ── Tab 2: Layer 2 ───────────────────────────────────────────

with tab2:
    st.subheader("Layer 2: 双维度诊断矩阵（SHAP × 因果效应）")
    st.caption(f"自适应阈值 | SHAP重要性: {shap_threshold:.3f} | 因果效应: {ce_threshold:.1f}分")

    dd_report, df_results, sv, ce_dict, _, _ = \
        layer2_dual_diagnosis(model, sample_student, df, student_id, X_train)

    # 散点图
    fig = plot_dual_diagnosis(
        sample_student, sv, ce_dict, student_id,
        shap_threshold, ce_threshold
    )
    st.pyplot(fig)

    # 诊断结果表格
    st.divider()
    st.write("**诊断明细**")
    display_df = df_results.rename(columns={
        '维度': '维度',
        '得分': '当前得分',
        'SHAP': 'SHAP值',
        '因果效应': '因果效应(分)',
        '分类': '诊断分类',
        '建议': '干预建议',
    })
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    # 分类摘要
    st.divider()
    st.write("**📊 诊断摘要**")
    categories = {'✅': [], '⚠️': [], '💡': [], '➖': []}
    for _, row in df_results.iterrows():
        categories[row['分类'].split()[0]].append(row['维度'])

    cats_cols = st.columns(4)
    with cats_cols[0]:
        if categories['✅']:
            names = [DIM_NAMES[d] for d in categories['✅']]
            st.success(f"✅ 优先干预\n\n" + "\n".join(f"- {n}" for n in names))
    with cats_cols[1]:
        if categories['⚠️']:
            names = [DIM_NAMES[d] for d in categories['⚠️']]
            st.warning(f"⚠️ 仅观察\n\n" + "\n".join(f"- {n}" for n in names))
    with cats_cols[2]:
        if categories['💡']:
            names = [DIM_NAMES[d] for d in categories['💡']]
            st.info(f"💡 潜在有效\n\n" + "\n".join(f"- {n}" for n in names))
    with cats_cols[3]:
        if categories['➖']:
            names = [DIM_NAMES[d] for d in categories['➖']]
            st.text(f"➖ 无需关注\n\n" + "\n".join(f"- {n}" for n in names))

# ── Tab 3: Layer 3 ───────────────────────────────────────────

with tab3:
    st.subheader("Layer 3: 个体干预效应估计（ITE）")
    st.caption(
        "此模块需要准实验数据（前测→干预→后测）。"
        "当前使用模拟数据演示闭环流程。"
    )

    n_exp = st.slider("模拟实验人数", 30, 200, 60, 10, key="exp_n")
    true_effect = st.slider("模拟真实干预效应（分）", 1.0, 15.0, 5.0, 0.5, key="exp_eff")

    if st.button("🔬 运行 ITE 估计"):
        exp_df = simulate_intervention_data(n=n_exp, true_effect=true_effect)
        X_exp = exp_df[CAUSAL_ORDER]
        T_exp = exp_df['T']
        Y_exp = exp_df['Y']

        results, summary = layer3_estimate_ite(X_exp, T_exp, Y_exp, method='xlearner')

        # 实验数据概况
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("实验组人数", f"{(T_exp==1).sum()}人")
        col_b.metric("对照组人数", f"{(T_exp==0).sum()}人")
        col_c.metric("实验组平均增长", f"{exp_df[T_exp==1]['Y'].mean():.2f}分")
        col_d.metric("对照组平均增长", f"{exp_df[T_exp==0]['Y'].mean():.2f}分")

        st.divider()

        # ITE 结果
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("X-Learner ATE", f"{summary['ate']:.2f}分")
        col2.metric("ITE 标准差", f"{summary['std']:.2f}分")
        col3.metric("正向效应", f"{summary['strong']}/{summary['positive']}人")
        col4.metric("负向效应", f"{summary['negative']}人")

        st.divider()

        # 闭环验证
        error = abs(summary['ate'] - true_effect)
        if error < 1.0:
            st.success(
                f"✅ 闭环验证通过！估计误差 {error:.2f}分，"
                f"X-Learner 准确还原了真实干预效应"
            )
        else:
            st.warning(
                f"⚠️ 估计误差 {error:.2f}分，样本量增大后可能改善"
            )

        st.dataframe(results.head(10), hide_index=True, use_container_width=True)

# ── Tab 4: 教师报告 ──────────────────────────────────────────

with tab4:
    st.subheader("给教师的诊断报告")

    if st.button("📝 生成报告", type="primary", use_container_width=True):
        with st.spinner("正在生成报告...（如果Ollama未启动将自动使用规则模板）"):
            report = generate_teacher_report(
                student_id, sample_student, sv, ce_dict,
                df_results, model, cf_best_dim, cf_best_gain,
                shap_threshold, ce_threshold
            )
        st.markdown(report)

    st.divider()
    st.caption(
        "💡 提示：启动本地 Ollama 服务可获取 LLM 生成的个性化报告；"
        "未启动时自动降级为规则模板报告。"
    )

# ── 页脚 ──────────────────────────────────────────────────────

st.divider()
st.caption(
    "三层因果可解释诊断系统 v5 · "
    "基于 XGBoost + SHAP + 后门调整 · "
    "Streamlit 部署版本"
)
