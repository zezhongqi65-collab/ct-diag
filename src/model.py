"""模型训练与保存"""

import os
import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.config import CAUSAL_ORDER


def train_model(df):
    """训练XGBoost回归模型"""
    features = CAUSAL_ORDER
    X = df[features]
    y = df['作业质量']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        reg_alpha=0.1, reg_lambda=1.0,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    return model, X_train, X_test, rmse


def save_model(model, path):
    """保存模型到文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """从文件加载模型"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")
    return joblib.load(path)
