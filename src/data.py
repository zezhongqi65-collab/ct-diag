"""数据准备与模拟"""

import numpy as np
import pandas as pd
from src.config import CAUSAL_ORDER, FEATURE_COLUMNS


def prepare_simulated_data(n=200):
    """生成模拟学生数据用于演示"""
    np.random.seed(42)
    df = pd.DataFrame({
        'A1': np.random.randint(1, 6, n), 'A2': np.random.randint(1, 6, n),
        'A3': np.random.randint(1, 6, n), 'A4': np.random.randint(1, 6, n),
        'A5': np.random.randint(1, 6, n), 'A6': np.random.randint(1, 6, n),
        'A7': np.random.randint(1, 6, n), 'A8': np.random.randint(1, 6, n),
        'A9': np.random.randint(1, 6, n), 'A10': np.random.randint(1, 6, n),
        'A11': np.random.randint(1, 6, n), 'A12': np.random.randint(1, 6, n),
        'A13': np.random.randint(1, 6, n), 'A14': np.random.randint(1, 6, n),
        'A15': np.random.randint(1, 6, n),
    })
    df['抽象'] = df[['A4', 'A5', 'A6']].mean(axis=1)
    df['分解'] = df[['A1', 'A2', 'A3']].mean(axis=1)
    df['算法设计'] = df[['A9', 'A10']].mean(axis=1)
    df['建模'] = df[['A7', 'A8']].mean(axis=1)
    df['评估'] = df[['A11', 'A12', 'A13', 'A14', 'A15']].mean(axis=1)
    df['作业质量'] = (
        df['抽象'] * 8 + df['分解'] * 4 + df['算法设计'] * 6 +
        df['建模'] * 5 + df['评估'] * 3 +
        2 * df['抽象'] * df['建模'] +
        np.random.normal(0, 5, n)
    ).clip(20, 100).round(1)
    return df


def prepare_from_csv(csv_path):
    """从教师上传的CSV读取数据，计算5维度均值"""
    df = pd.read_csv(csv_path)
    for dim, cols in FEATURE_COLUMNS.items():
        available = [c for c in cols if c in df.columns]
        if not available:
            raise ValueError(f"缺少{dim}维度的指标列: {cols}")
        if len(available) < len(cols):
            print(f"   ⚠ {dim}维度: 使用{available}计算均值（部分列缺失）")
        df[dim] = df[available].mean(axis=1)

    # 如果没有作业质量列，提醒教师需要提供
    if '作业质量' not in df.columns:
        raise ValueError(
            "CSV缺少'作业质量'列。"
            "如需使用诊断功能，请包含此列。若只想查看指标得分，可先使用模拟数据模式。"
        )
    return df


def simulate_intervention_data(n=60, true_effect=5.0):
    """生成准实验干预数据"""
    np.random.seed(123)
    df = pd.DataFrame({
        '抽象': np.random.uniform(1, 5, n),
        '分解': np.random.uniform(1, 5, n),
        '算法设计': np.random.uniform(1, 5, n),
        '建模': np.random.uniform(1, 5, n),
        '评估': np.random.uniform(1, 5, n),
    })
    df['T'] = np.random.binomial(1, 0.5, n)
    df['baseline'] = (
        df['抽象'] * 8 + df['分解'] * 4 + df['建模'] * 5 +
        np.random.normal(0, 5, n)
    )
    df['post'] = df['baseline'] + df['T'] * true_effect + np.random.normal(0, 3, n)
    df['Y'] = df['post'] - df['baseline']
    return df
