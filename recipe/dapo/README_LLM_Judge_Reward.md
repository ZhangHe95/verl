# LLM-as-a-Judge Reward Model for VERL

This implementation provides an LLM-based reward model that generates natural language evaluations and extracts numerical scores for use in VERL (Versatile Efficient Reinforcement Learning) training.

## 概述

传统的reward model通常是专门训练的神经网络，输出单一的数值分数。而LLM-as-a-Judge方法使用大语言模型生成详细的文本评价，然后从中提取分数。这种方法的优势包括：

- **更丰富的反馈**：生成详细的评价文本，不仅有分数还有理由
- **更强的可解释性**：可以看到模型为什么给出特定分数
- **灵活性**：可以轻松调整评价标准和prompt模板
- **无需专门训练**：直接使用现有的强大LLM

## 文件说明

- `llm_judge_reward.py`: 核心实现文件
- `ppo_with_llm_judge_config.yaml`: VERL训练配置文件
- `run_ppo_llm_judge.sh`: 训练运行脚本
- `test_llm_judge.py`: 功能测试脚本

## 快速开始

### 1. 安装依赖

```bash
# 安装VERL和相关依赖
pip install verl
pip install vllm
pip install transformers
pip install torch
```

### 2. 测试LLM Judge功能

首先运行测试脚本确保一切正常：

```bash
python test_llm_judge.py
```

这个脚本会测试：
- 基本的LLM评价功能
- VERL兼容接口
- 批处理性能

### 3. 配置训练参数

编辑 `ppo_with_llm_judge_config.yaml` 文件，主要需要修改：

```yaml
# 数据路径
data:
  train_files: ["path/to/your/train.parquet"]
  val_files: ["path/to/your/val.parquet"]

# 模型路径
actor_rollout_ref:
  model:
    path: "your/actor/model/path"

# Judge模型配置
custom_reward_function:
  reward_kwargs:
    judge_model_path: "your/judge/model/path"
    # 可以使用与actor相同或不同的模型
```

### 4. 运行训练

```bash
# 修改脚本中的数据路径
vim run_ppo_llm_judge.sh

# 运行训练
bash run_ppo_llm_judge.sh
```

## 核心功能

### LLMJudgeReward 类

主要的reward model类，具有以下功能：

```python
from llm_judge_reward import LLMJudgeReward

# 初始化judge
judge = LLMJudgeReward(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    use_vllm=True,  # 使用vLLM加速推理
    temperature=0.1,  # 低温度获得更稳定的评价
    max_eval_length=512  # 评价文本最大长度
)

# 批量评价
prompts = ["User question 1", "User question 2"]
responses = ["Model response 1", "Model response 2"]
scores, evaluations = judge.compute_scores(prompts, responses, return_evaluations=True)
```

### 自定义评价标准

你可以自定义prompt模板来改变评价标准：

```python
custom_template = """
You are an expert evaluator. Please evaluate the AI response based on:
1. Technical accuracy
2. Completeness
3. Clarity of explanation
4. Practical usefulness

User Query: {prompt}
AI Response: {response}

Provide detailed feedback and rate from 0-10:
Feedback: [Your analysis]
Score: [0-10 rating]
"""

judge = LLMJudgeReward(
    model_path="your/model/path",
    judge_prompt_template=custom_template
)
```

## 高级配置

### 1. 多GPU配置

```yaml
# 在配置文件中设置
custom_reward_function:
  reward_kwargs:
    tensor_parallel_size: 2  # 使用2个GPU
    gpu_memory_utilization: 0.4  # 每个GPU使用40%内存
```

### 2. 分数提取模式

可以自定义从评价文本中提取分数的正则表达式：

```python
judge = LLMJudgeReward(
    model_path="your/model/path",
    score_extraction_pattern=r"Rating:\s*(\d+(?:\.\d+)?)"  # 匹配 "Rating: 8.5"
)
```

### 3. 异步执行

在VERL配置中启用异步执行以提高性能：

```yaml
reward_model:
  launch_reward_fn_async: True  # 异步执行reward函数
```

## 性能优化建议

### 1. 使用vLLM

vLLM比HuggingFace Transformers快很多：

```python
judge = LLMJudgeReward(
    model_path="your/model/path",
    use_vllm=True,  # 推荐使用vLLM
    tensor_parallel_size=2  # 根据GPU数量调整
)
```

### 2. 合理设置batch size

在VERL配置中调整batch size：

```yaml
data:
  train_batch_size: 256  # 根据GPU内存调整
actor_rollout_ref:
  actor:
    ppo_micro_batch_size_per_gpu: 4  # 小一些避免内存不足
```

### 3. 控制评价长度

```python
judge = LLMJudgeReward(
    model_path="your/model/path",
    max_eval_length=256,  # 较短的评价文本
    temperature=0.1  # 低温度减少随机性
)
```

## 故障排除

### 1. 内存不足

如果遇到GPU内存不足：

```yaml
# 减少GPU内存使用
custom_reward_function:
  reward_kwargs:
    gpu_memory_utilization: 0.3  # 减少到30%
    tensor_parallel_size: 1  # 使用单GPU

# 减少batch size
actor_rollout_ref:
  actor:
    ppo_micro_batch_size_per_gpu: 2
```

### 2. 分数提取失败

如果无法正确提取分数，检查LLM的输出格式：

```python
# 添加调试信息
import logging
logging.basicConfig(level=logging.INFO)

# 或者查看评价文本
scores, evaluations = judge.compute_scores(prompts, responses, return_evaluations=True)
for eval_text in evaluations:
    print(f"Evaluation: {eval_text}")
```

### 3. 推理速度慢

- 使用vLLM而不是HuggingFace
- 增加`tensor_parallel_size`
- 减少`max_eval_length`
- 使用更小的judge模型

## 自定义评价函数

你也可以实现完全自定义的评价逻辑：

```python
def custom_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # 你的自定义逻辑
    if "math" in data_source:
        # 数学问题的特殊评价逻辑
        pass
    elif "code" in data_source:
        # 代码问题的特殊评价逻辑
        pass
    
    # 调用LLM judge
    from llm_judge_reward import compute_score
    return compute_score(data_source, solution_str, ground_truth, extra_info)
```

## 贡献和反馈

如果你在使用过程中遇到问题或有改进建议，欢迎：
- 提交Issue报告问题
- 分享你的评价prompt模板
- 贡献性能优化建议

## 示例输出

以下是LLM Judge的典型输出示例：

```
User Query: What is the capital of France?
AI Response: The capital of France is Paris.

Generated Evaluation:
The response correctly identifies Paris as the capital of France. The answer is accurate, direct, and answers the question completely. However, it could be enhanced with additional context about Paris being a major cultural and economic center.

Evaluation: The response is factually correct and directly answers the question. It's concise and clear.
Score: 8

Extracted Score: 0.8
```

通过这种方式，你不仅得到了数值分数，还获得了详细的评价理由，有助于理解模型的表现并进行进一步优化。 