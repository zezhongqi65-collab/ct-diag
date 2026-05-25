"""可视化工具"""

import matplotlib
import matplotlib.font_manager as fm

# 清除字体缓存，确保新装字体能被发现
fm._load_fontmanager(try_read_cache=False)

# 跨平台中文字体：Windows → Linux → macOS → 回退
_CJK_FONTS = [
    'Microsoft YaHei', 'SimHei',           # Windows
    'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',  # Linux (Streamlit Cloud)
    'PingFang SC', 'Heiti SC',              # macOS
    'Noto Sans CJK SC', 'Noto Sans SC',     # 通用
    'sans-serif',                            # 最终回退
]
matplotlib.rcParams['font.sans-serif'] = _CJK_FONTS
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.config import CAUSAL_ORDER, DIM_NAMES


def plot_dual_diagnosis(student, sv, ce_dict, student_id,
                        shap_threshold, ce_threshold):
    """绘制 SHAP × 因果效应 双维度散点图"""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors_map = {
        '✅': '#2ECC71', '⚠️': '#F39C12',
        '💡': '#3498DB', '➖': '#95A5A6',
    }

    for i, feat in enumerate(CAUSAL_ORDER):
        s = abs(sv[i])
        c = ce_dict[feat]
        if s > shap_threshold and c > ce_threshold:
            color, size = colors_map['✅'], 300
        elif s > shap_threshold and c <= ce_threshold:
            color, size = colors_map['⚠️'], 250
        elif s <= shap_threshold and c > ce_threshold:
            color, size = colors_map['💡'], 200
        else:
            color, size = colors_map['➖'], 150
        ax.scatter(s, c, s=size, color=color, alpha=0.8,
                   edgecolors='white', linewidth=2, zorder=3)
        ax.annotate(DIM_NAMES[feat], (s, c),
                    textcoords="offset points", xytext=(10, 5), fontsize=11)

    ax.axhline(y=ce_threshold, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=shap_threshold, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('|SHAP值|（模型归因重要性）', fontsize=13)
    ax.set_ylabel('因果效应（分）', fontsize=13)
    ax.set_title(
        f'{student_id} 双维度诊断散点图（SHAP × 因果效应）\n'
        f'阈值基于训练集自适应计算', fontsize=13, fontweight='bold'
    )

    legend_elements = [
        Patch(facecolor=colors_map['✅'], label='✅ 优先干预'),
        Patch(facecolor=colors_map['⚠️'], label='⚠️ 仅观察'),
        Patch(facecolor=colors_map['💡'], label='💡 潜在有效'),
        Patch(facecolor=colors_map['➖'], label='➖ 无需关注'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
