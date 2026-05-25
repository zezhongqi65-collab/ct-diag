"""Layer 3: 个体干预效应估计（T-Learner / X-Learner）"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from src.config import CAUSAL_ORDER


def tlearner_ite(X, T, Y):
    """T-Learner 估计个体干预效应"""
    model_0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_0.fit(X[T == 0], Y[T == 0])
    model_1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_1.fit(X[T == 1], Y[T == 1])
    return model_1.predict(X) - model_0.predict(X)


def xlearner_ite(X, T, Y):
    """X-Learner 估计个体干预效应"""
    model_0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_0.fit(X[T == 0], Y[T == 0])
    model_1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_1.fit(X[T == 1], Y[T == 1])
    D = np.zeros(len(Y))
    D[T == 1] = Y[T == 1] - model_0.predict(X[T == 1])
    D[T == 0] = model_1.predict(X[T == 0]) - Y[T == 0]
    tau_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    tau_model.fit(X, D)
    return tau_model.predict(X)


def layer3_estimate_ite(X, T, Y, method='xlearner'):
    """Layer 3: 估计个体干预效应（ITE）"""
    if method == 'tlearner':
        ite = tlearner_ite(X, T, Y)
    elif method == 'xlearner':
        ite = xlearner_ite(X, T, Y)
    else:
        raise ValueError("method必须是 'tlearner' 或 'xlearner'")

    results = pd.DataFrame({
        'ITE': ite,
        '干预建议': np.where(
            ite > 3, '✅ 强烈推荐（预计提升>3分）',
            np.where(ite > 0, '💡 可能有效（预计提升0-3分）',
                     np.where(ite > -3, '➖ 效果不显著', '⚠️ 可能有害'))
        )
    })

    summary = {
        'method': method.upper(),
        'ate': ite.mean(),
        'std': ite.std(),
        'positive': int((ite > 0).sum()),
        'strong': int((ite > 3).sum()),
        'negative': int((ite < 0).sum()),
    }
    return results, summary
