# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
anaconda 路径：C:\ProgramData\anaconda3\envs\Model

## 项目概述

这是一个基于约束解码（constrained decoding）的**中国古典诗词生成系统**。核心思路是在大语言模型逐 token 生成时，通过 HuggingFace `LogitsProcessor` 接口实时干预 logits 分布，强制模型输出符合特定词牌/诗体格律（平仄、押韵、字数、句读）的文本。

## 核心架构

```
main.py                    # 入口：模型加载 → Prompt 构造 → 生成循环
  ├── data_manager.py      # 韵书 + 格律 JSON 加载，提供平仄/韵部查询
  ├── vocab_indexer.py     # 离线：遍历 tokenizer 词表，建立 (字数,平仄) + 韵部 → token_id 索引
  ├── state_machine.py     # 运行时：跟踪当前阕/句/字位置，决定下一步允许的平仄模式
  └── logits_processor.py  # 运行时：每步调用 __call__，根据 state_machine 的允许集掩码 logits
```

**关键数据流：**
1. `DataManager` 加载韵书 JSON（`Rhyme/`）和格律 JSON（`Meter/`）
2. `VocabIndexer` 离线遍历 tokenizer 全部词表，为每个中文 token 预计算平仄序列和韵部，存入 `pattern_tokens` 和 `rhyme_tokens` 字典
3. 生成时 `ConstraintLogitsProcessor.__call__()` 每步执行：
   - 解码已生成的 token，送入 `state_machine.advance_state()` 更新位置
   - 调用 `state_machine.get_allowed_patterns()` 获取当前可用的 (字数, 平仄模式, 韵部要求) 列表
   - 从 `vocab_indexer` 查找合法 token_id 集合
   - 将所有非法 token 的 logit 设为 `-inf`，合法 token 保留原始分数
   - 对已出现过的字应用指数衰减重复惩罚

## 命令

```bash
# 安装依赖 (注意：bitsandbytes 需额外安装)
pip install -r requirements.txt

# 运行主程序（诗词生成）
python main.py

# 调试词表索引构建
python debug_vocab.py
```

所有配置集中在 `main.py:main()` 函数开头的"集中配置区域"：
- `model_name`：模型路径（支持 Qwen3-4B、Llama-3.2-3B、DeepSeek-R1-Distill-Qwen-1.5B）
- `meter_type`：`"宋词"` 或 `"唐诗"`（决定加载 `songci.json` 还是 `TongPoem.json`）
- `rhyme_dict_name`：`"Cilin"`（词林正韵）、`"Pinshui"`（平水韵）、`"Xinyun"`（中华新韵）
- `task_type`：`"instruction"` / `"zero-shot"` / `"one-shot"` / `"completion"`
- `use_constraints`：`True` 启用约束解码，`False` 做无约束对比
- `use_thinking`：DeepSeek R1 系列必须设为 `True`

## 关键文件

- **Meter/songci.json** — 宋词格律定义（每个词牌含阕数、每句字数和平仄模式、押韵位置和韵部编号）
- **Meter/TongPoem.json** — 唐诗格律定义（五律、七律等）
- **Rhyme/Cilin.json** — 词林正韵（韵部 → 声调 → 字列表）
- **Rhyme/Pinshui.json** — 平水韵
- **Rhyme/Xinyun.json** — 中华新韵
- **output/** — 生成结果保存目录

## 格律 JSON 格式约定

每个词牌/诗体的 JSON 对象结构：
```json
{
  "rhyme_type": "平韵",
  "number_of_stanzas": 2,
  "stanza1": {
    "num_lines": 5,
    "lines": ["中仄/平平/仄", "平平/中仄/平", ...],
    "rhyme_1_positions": [2, 5],
    "rhyme_2_positions": [7]
  }
}
```
- `lines` 中使用 `中` 表示可平可仄，`/` 表示句中节奏停顿，`、` 表示必须输出顿号
- `rhyme_X_positions` 中的 `X` 为韵部编号，对应多个韵部时需要保持同韵部内的字押韵一致

## 技术要点

- `GenerationStateMachine` 在每次生成开始时必须**重新初始化**，因为其内部包含韵部锁定等有状态信息
- 唐诗的重复字检测由 `tang_logits_processor.py` 专门处理，采用分层检测策略（句内重字扣分、n-gram 硬拒绝等），不再使用旧版 `-inf` 一字封杀
- BPE tokenizer（如 Qwen）的词表 token 是字节编码，需要用 `tokenizer.decode([id])` 获取真实字符，不能直接读取 token 字符串
- `[title]` 和 `[content]` 标记是程序解析生成进度的关键依赖——在遇到 `[content]` 之前约束逻辑不启动
