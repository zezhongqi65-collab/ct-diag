"""Layer 1: 反事实分析（Counterfactual Analysis）"""

import pandas as pd
import numpy as np
from src.config import CAUSAL_ORDER, DIM_NAMES


def counterfactual_simulation(model, student, feature, value_range=None):
    """对单个维度进行反事实模拟"""
    if value_range is None:
        value_range = [1, 2, 3, 4, 5]
    base_pred = model.predict(student.values.reshape(1, -1))[0]
    current_val = student[feature]
    results = []
    for val in value_range:
        modified = student.copy()
        modified[feature] = val
        new_pred = model.predict(modified.values.reshape(1, -1))[0]
        results.append({
            f'{feature}假设值': val,
            '预测成绩': round(new_pred, 1),
            '相比当前变化': round(new_pred - base_pred, 1),
            '是否当前值': '★ 当前' if abs(val - current_val) < 0.1 else ''
        })
    return pd.DataFrame(results), base_pred, current_val


def layer1_counterfactual(model, student, student_id):
    """Layer 1 完整分析：各维度的反事实模拟"""
    features = CAUSAL_ORDER
    report_lines = []
    report_lines.append(f"\n{'='*60}")
    report_lines.append(f"【Layer 1: 反事实分析（CFE）】学生 {student_id}")
    report_lines.append(f"{'='*60}")
    report_lines.append('\n⚠️ 免责声明：以下结果为模型层面的假设模拟（CFE），')
    report_lines.append('   仅展示「若该维度独立变化，模型预测如何改变」。')
    report_lines.append('   实际干预效果请以Layer 2因果效应为准。')

    base_pred = model.predict(student.values.reshape(1, -1))[0]
    report_lines.append(f"\n{'─'*50}")
    report_lines.append(f"当前预测作业质量: {base_pred:.1f}分/100分")
    report_lines.append(f"\n{'─'*50}")
    report_lines.append("各维度提升潜力模拟（固定其他维度不变）：")
    report_lines.append(f"{'─'*50}")

    best_dim = None
    best_gain = -999
    all_dfs = {}

    for feat in features:
        df_cf, _, current_val = counterfactual_simulation(model, student, feat)
        all_dfs[feat] = df_cf
        target_row = df_cf[df_cf[f'{feat}假设值'] == 5]
        gain = target_row['相比当前变化'].values[0] if not target_row.empty else 0
        report_lines.append(f"\n【{DIM_NAMES[feat]}】当前{current_val:.1f}分")
        report_lines.append(df_cf.to_string(index=False))
        if gain > best_gain:
            best_gain = gain
            best_dim = feat

    report_lines.append(f"\n{'='*60}")
    report_lines.append("【反事实结论（模型模拟，非真实因果）】")
    if best_dim:
        report_lines.append(f"  优先提升维度: {DIM_NAMES[best_dim]}")
        report_lines.append(f"  若从当前{student[best_dim]:.1f}分 → 5分")
        report_lines.append(
            f"  模型预测变化: {base_pred:.1f}分 → "
            f"{base_pred + best_gain:.1f}分 (+{best_gain:.1f}分)"
        )
        report_lines.append("  ⚠️ 注意: 以上为模型假设，真实效果请以Layer 2因果效应为准")
    report_lines.append(f"{'='*60}")

    return '\n'.join(report_lines), best_dim, best_gain, all_dfs
