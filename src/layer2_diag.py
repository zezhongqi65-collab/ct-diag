"""Layer 2: 双维度诊断矩阵（SHAP × 因果效应）"""

import numpy as np
import pandas as pd
import shap

try:
    import statsmodels.api as sm
except ImportError:
    sm = None

from src.config import CAUSAL_ORDER, DAG_EDGES, DIM_NAMES


# ── 因果效应估计 ─────────────────────────────────────────────

def backdoor_causal_effect(df, treatment, outcome, confounders=None):
    """后门调整估计因果效应"""
    if sm is not None and confounders and len(confounders) > 0:
        formula = f"{outcome} ~ {treatment} + " + " + ".join(confounders)
        try:
            fit = sm.OLS.from_formula(formula, data=df).fit()
            ate = fit.params[treatment]
            p_value = fit.pvalues[treatment]
            # p>0.05 时不归零，改为衰减为 30%
            if p_value > 0.05:
                return ate * 0.3
            return ate
        except Exception:
            pass

    # 回退方案：分组比较
    if confounders and len(confounders) > 0:
        effects = []
        for _, group in df.groupby(confounders):
            high = group[group[treatment] >= 4][outcome]
            low = group[group[treatment] < 3][outcome]
            if len(high) > 2 and len(low) > 2:
                effects.append(high.mean() - low.mean())
        return np.mean(effects) if effects else 0.0
    else:
        high = df[df[treatment] >= 4][outcome]
        low = df[df[treatment] < 3][outcome]
        return (high.mean() - low.mean()) if len(high) > 2 and len(low) > 2 else 0.0


def estimate_all_causal_effects(df):
    """估计所有维度的因果效应"""
    effects = {}
    for feat in CAUSAL_ORDER:
        parents = [s for s, t in DAG_EDGES if t == feat]
        effects[feat] = backdoor_causal_effect(df, feat, '作业质量', parents)
    return effects


def trace_root_cause(dim, student):
    """追溯根因维度"""
    parents = [s for s, t in DAG_EDGES if t == dim]
    if not parents:
        return None
    weak_parents = [p for p in parents if student[p] < 3.0]
    return weak_parents if weak_parents else None


# ── 自适应阈值 ────────────────────────────────────────────────

def compute_adaptive_thresholds(model, X_train):
    """计算自适应阈值"""
    explainer = shap.TreeExplainer(model)
    sv_all = np.abs(explainer.shap_values(X_train))
    shap_threshold = np.percentile(sv_all.mean(axis=0), 50)
    ce_all = estimate_all_causal_effects(
        X_train.assign(作业质量=model.predict(X_train))
    )
    ce_threshold = np.median(list(ce_all.values()))
    return shap_threshold, ce_threshold


# ── 双维度诊断 ────────────────────────────────────────────────

def layer2_dual_diagnosis(model, student, df_all, student_id, X_train):
    """Layer 2 完整分析：双维度诊断矩阵"""
    features = CAUSAL_ORDER
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(student.values.reshape(1, -1))[0]
    ce_dict = estimate_all_causal_effects(df_all)
    shap_threshold, ce_threshold = compute_adaptive_thresholds(model, X_train)

    report_lines = []
    report_lines.append(f"\n{'='*60}")
    report_lines.append(f"【Layer 2: 双维度诊断（SHAP × 因果效应）】学生 {student_id}")
    report_lines.append(f"{'='*60}")
    report_lines.append(
        f"自适应阈值 | SHAP重要性: {shap_threshold:.3f} | 因果效应: {ce_threshold:.1f}分"
    )
    report_lines.append("（基于训练集分布自动计算，非固定值）")
    report_lines.append(f"\n{'─'*50}")

    categories = {'✅': [], '⚠️': [], '💡': [], '➖': []}
    results_list = []

    for i, feat in enumerate(features):
        shap_val = sv[i]
        ce_val = ce_dict[feat]
        shap_high = abs(shap_val) > shap_threshold
        ce_positive = ce_val > ce_threshold

        if shap_high and ce_positive:
            tag = '✅ 优先干预'
            action = f"立即安排{DIM_NAMES[feat]}专项训练（预计提升{ce_val:.1f}分）"
        elif shap_high and not ce_positive:
            tag = '⚠️ 仅观察'
            root = trace_root_cause(feat, student)
            if root:
                root_names = ','.join([DIM_NAMES[r] for r in root])
                action = (
                    f"SHAP显示重要但单独干预无效。根因可能在「{root_names}」，"
                    "建议先练根因维度"
                )
            else:
                action = "SHAP显示重要但干预无效，可能受其他未测量因素影响"
        elif not shap_high and ce_positive:
            tag = '💡 潜在有效'
            action = (
                f"模型未识别但干预可能有效，作为二级备选方案"
                f"（预计提升{ce_val:.1f}分）"
            )
        else:
            tag = '➖ 无需关注'
            action = "当前维度正常"

        categories[tag.split()[0]] = categories.get(tag.split()[0], []) + [feat]
        report_lines.append(f"\n【{DIM_NAMES[feat]}】")
        report_lines.append(
            f"  得分: {student[feat]:.1f}/5 | SHAP: {shap_val:+.3f} "
            f"| 因果效应: +{ce_val:.1f}分"
        )
        report_lines.append(f"  分类: {tag}")
        report_lines.append(f"  建议: {action}")
        results_list.append({
            '维度': feat, '得分': student[feat],
            'SHAP': shap_val, '因果效应': ce_val,
            '分类': tag, '建议': action,
        })

    report_lines.append(f"\n{'─'*50}")
    report_lines.append("\n📊 诊断摘要:")
    if categories['✅']:
        names = ', '.join([DIM_NAMES[d] for d in categories['✅']])
        report_lines.append(f"   ✅ 优先干预: {names}")
    if categories['⚠️']:
        names = ', '.join([DIM_NAMES[d] for d in categories['⚠️']])
        report_lines.append(f"   ⚠️ 仅观察: {names}")
    if categories['💡']:
        names = ', '.join([DIM_NAMES[d] for d in categories['💡']])
        report_lines.append(f"   💡 潜在有效: {names}")
    report_lines.append(f"{'='*60}")

    report = '\n'.join(report_lines)
    return report, pd.DataFrame(results_list), sv, ce_dict, shap_threshold, ce_threshold
