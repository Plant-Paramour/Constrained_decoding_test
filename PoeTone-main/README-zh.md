# PoeTone：基于大语言模型的结构化中文宋词约束生成框架

本仓库包含论文 **PoeTone: A Framework for Constrained Generation of Structured Chinese Songci with LLMs** 的代码实现。

## 环境与依赖

安装所需依赖：

```bash
pip install -r requirements.txt
```

## 代码与数据

用于评估大语言模型生成中文宋词能力，请运行 `evaluation/` 目录下的脚本：

- `llama3.py`
- `mistral.py`
- `deepseekr1.py`
- `qwen3.py`
- `gpt4o.py`

用于统一评测：

- `evaluation_all.py`

用于清洗生成的宋词：

- `raw_data_cleaning.py`

用于应用 generate-critic 方法微调大模型：

- `create_best_of_n_dataset.py`
- `run_sft.py`

