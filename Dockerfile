FROM python:3.11-slim

# 安装中文字体
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        fonts-wqy-zenhei \
        fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建模型目录
RUN mkdir -p models

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 健康检查
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 启动
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
