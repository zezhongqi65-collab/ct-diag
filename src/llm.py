"""LLM 集成：教师诊断报告生成（Ollama + 云端API降级）"""

import os
import requests
from src.config import DIM_NAMES, OLLAMA_MODEL


def ollama_generate(prompt, model=OLLAMA_MODEL, timeout=60):
    """调用本地 Ollama 生成文本"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.7, "num_predict": 800},
            },
            timeout=timeout
        )
        if response.status_code == 200:
            return response.json().get("response", "[Ollama生成失败]")
        return "[Ollama请求失败]"
    except requests.exceptions.ConnectionError:
        return None  # 返回 None 表示需要降级
    except requests.exceptions.Timeout:
        return f"[错误：Ollama请求超时（{timeout}秒）。请检查模型是否已加载。]"
    except Exception as e:
        return f"[错误：{str(e)}]"


def generate_fallback_report(student_id, student, df_results, base_pred,
                             cf_best_dim, cf_best_gain):
    """不依赖 LLM 的规则模板报告（Ollama不可用时的降级方案）"""
    rows = df_results.to_dict('records')

    # 找出优先干预维度
    priority = [r for r in rows if r['分类'] == '✅ 优先干预']
    observe = [r for r in rows if r['分类'] == '⚠️ 仅观察']
    potential = [r for r in rows if r['分类'] == '💡 潜在有效']

    lines = []
    lines.append("本报告基于可解释AI分析，SHAP值反映模型归因，因果效应反映干预预期效果。")
    lines.append("（当前为离线规则模板，启动Ollama后可获得LLM生成的详细报告。）")
    lines.append("")
    lines.append(f"## 学生 {student_id} 诊断报告")
    lines.append(f"- 预测作业质量：{base_pred:.1f}分/100分")
    lines.append("")

    # 核心问题
    lines.append("### 一、核心问题")
    if priority:
        names = [DIM_NAMES[r['维度']] for r in priority]
        lines.append(f"该学生最需优先干预的维度为：**{'、'.join(names)}**。")
        for r in priority:
            lines.append(
                f"- {DIM_NAMES[r['维度']]}：当前{r['得分']:.1f}分，"
                f"干预预期提升{r['因果效应']:.1f}分"
            )
    else:
        lines.append("该学生各维度发展较为均衡，无明显的优先干预维度。")
    lines.append("")

    # 根因分析
    if observe:
        lines.append("### 二、根因分析")
        for r in observe:
            lines.append(
                f"- **{DIM_NAMES[r['维度']]}**：{r['建议']}"
            )
        lines.append("")

    # 干预建议
    lines.append("### 三、干预建议")
    lines.append("1. **第一步**：针对优先干预维度的专项训练，每周2-3次，每次20-30分钟")
    lines.append('2. **第二步**：通过根因维度训练，间接提升被标记为「仅观察」的维度')
    lines.append("3. **第三步**：4周后进行后测，评估干预效果")
    lines.append("4. **长期**：建立学生个人计算思维档案，持续追踪5个维度的发展轨迹")
    lines.append("")

    # 预期效果
    lines.append("### 四、预期效果")
    positive = [r for r in rows if r['因果效应'] > 0]
    if positive:
        total_gain = sum(r['因果效应'] for r in positive)
        lines.append(f"若全部正效应维度均得到有效干预，预计总分可提升约{total_gain:.1f}分。")
        lines.append("建议重点关注因果效应最大的1-2个维度，集中资源实现最大化提升。")

    return '\n'.join(lines)


def generate_teacher_report(student_id, student, sv, ce_dict, df_results, model,
                            cf_best_dim, cf_best_gain, shap_threshold, ce_threshold):
    """生成教师诊断报告（优先 LLM，不可用时降级为规则模板）"""
    base_pred = model.predict(student.values.reshape(1, -1))[0]

    # 构建维度明细
    dim_lines = []
    for _, row in df_results.iterrows():
        dim_lines.append(
            f"- {DIM_NAMES[row['维度']]}: 得分{row['得分']:.1f}/5, "
            f"SHAP={row['SHAP']:+.3f}, 因果效应+{row['因果效应']:.1f}分, "
            f"分类：{row['分类']}"
        )
    dim_lines_str = '\n'.join(dim_lines)

    prompt = f"""你是一位资深信息技术教育专家和教学诊断顾问。请根据以下学生的双维度AI诊断数据，为任课教师撰写一段专业、具体、可操作的诊断报告与干预建议（350字以内）。

## 学生诊断数据

- 学生编号：{student_id}
- 预测作业质量：{base_pred:.1f}分/100分

### 反事实分析（Layer 1，模型模拟，仅供参考）
- 优先提升维度：{DIM_NAMES.get(cf_best_dim, '无')}
- 若该维度提升至满分，模型预测变化：+{cf_best_gain:.1f}分
- ⚠️ 注意：反事实结果为模型假设模拟，实际干预效果请以Layer 2因果效应为准

### 双维度分析（Layer 2：SHAP × 因果效应，基于DAG后门调整）
- 自适应阈值 | SHAP: {shap_threshold:.3f} | 因果效应: {ce_threshold:.1f}分
{dim_lines_str}

## 报告要求

1. 核心问题：明确指出该学生最需优先干预的1-2个维度
2. 根因分析：如果某个维度被标记为"⚠️仅观察"，分析其根因维度
3. 干预建议：给出3-4条具体的、可在课堂实施的干预策略
4. 预期效果：基于因果效应值（非反事实模拟），量化干预后的预期提升
5. 优先级排序：明确告诉教师"第一步做什么、第二步做什么"
6. 语言专业但通俗易懂，适合中学信息技术教师直接参考
7. 报告开头加一句："本报告基于可解释AI分析，SHAP值反映模型归因，因果效应反映干预预期效果"

请直接输出诊断报告正文。"""

    result = ollama_generate(prompt)
    if result is None:
        # Ollama 不可用，使用规则模板降级
        return generate_fallback_report(
            student_id, student, df_results, base_pred,
            cf_best_dim, cf_best_gain
        )
    return result
