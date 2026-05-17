# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
anaconda 路径：C:\ProgramData\anaconda3\envs\Model

## 项目概述

这是一个基于约束解码（constrained decoding）的**中国古典诗词生成系统**。核心思路是在大语言模型逐 token 生成时，通过 HuggingFace `LogitsProcessor` 接口实时干预 logits 分布，强制模型输出符合特定词牌/诗体格律（平仄、押韵、字数、句读）的文本。

## 双轨架构：宋词 + 唐诗

项目支持两种诗体，各自有独立的状态机和 LogitsProcessor，在 `main.py` 中通过 `meter_type` 配置切换：

```
                    ┌── DataManager（韵书 + 格律 JSON）
                    │
                    ├── VocabIndexer（离线遍历词表 → pattern_tokens + rhyme_tokens）
                    │
        ┌───────────┴───────────┐
        │                       │
   【宋词路径】              【唐诗路径】
   meter_type="宋词"         meter_type="唐诗"
        │                       │
   GenerationStateMachine    TangPoemStateMachine
   (JSON 驱动，依赖            (纯算法驱动，参数：
   Songci_Meter/ 格律)        line_length + num_lines，
                              不读格律 JSON)
        │                       │
   ConstraintLogitsProcessor TangPoemLogitsProcessor
   (标点/顿号/换行分阶段       (集成 poem_verifier 风格
    掩码 + 指数衰减重复惩罚)    的 _verifier_check 两级
                              兜底 + 二四六分明/孤平/
                              三连同/押韵一致性)
```

**关键差异：**
- **宋词**：依赖 `Songci_Meter/` 格律定义（每词牌的句数、字数、平仄模式、韵部位置），状态机按 JSON 规则推进
- **唐诗**：不读格律 JSON，`main.py:parse_tang_format()` 从诗体名（如"七律"）解析参数（5/7 言、4/8 句），`TangPoemStateMachine` 纯算法生成二四六分明模式；`TangPoemLogitsProcessor._verifier_check()` 做细粒度逐 token 校验，若全部候选被拒则降级到 `_critical_checks()`（放宽次要规则但保留核心格律底线）

## 核心架构

```
main.py                    # 入口：模型加载 → Prompt 构造 → 生成循环
  ├── data_manager.py      # 韵书 + 格律 JSON 加载，提供平仄/韵部查询
  ├── vocab_indexer.py     # 离线：遍历 tokenizer 词表，建立 (字数,平仄) + 韵部 → token_id 索引
  ├── state_machine.py     # 宋词(GenerationStateMachine) + 唐诗(TangPoemStateMachine) 状态机
  └── logits_processor.py  # 宋词(ConstraintLogitsProcessor) + 唐诗(TangPoemLogitsProcessor) 约束干预器
```

**关键数据流：**
1. `DataManager` 加载韵书 JSON（`Rhyme/`）和格律 JSON（`Songci_Meter/`）
2. `VocabIndexer` 离线遍历 tokenizer 全部词表，为每个中文 token 预计算平仄序列和韵部，存入 `pattern_tokens` 和 `rhyme_tokens` 字典
3. 生成时 LogitsProcessor `__call__()` 每步执行：
   - 解码已生成的 token，送入 `state_machine.advance_state()` 更新位置
   - 调用 `state_machine.get_allowed_patterns()` 获取当前可用的 (字数, 平仄模式, 韵部要求) 列表
   - 从 `vocab_indexer` 查找合法 token_id 集合
   - 将所有非法 token 的 logit 设为 `-inf`，合法 token 保留原始分数
   - （宋词）对已出现过的字应用指数衰减重复惩罚
   - （唐诗）逐 token 调用 `_verifier_check()` 做完整格律校验，含多级兜底

## 评估体系

`evaluation/` 目录下有两套评估器，均基于 中华新韵（pypinyin）的 3 维评分体系（结构 40% + 平仄 30% + 押韵 30%）：

### evaluation/evaluate.py — 统一评估入口
支持宋词和唐诗的批量评估，命令行用法：
```bash
python evaluation/evaluate.py --meter Songci_Meter/ --input evaluation/evaluation_input --output evaluation/evaluation_output
```
- **SongciEvaluator**：读取 `Songci_Meter/` 格律模板，对比生成结果的字数/平仄/押韵
- **TangPoemEvaluator**：规则驱动（二四六分明/三连同/孤平/末字收束），不依赖模板
- 输入目录结构：`evaluation_input/Songci/*.txt` + `evaluation_input/Tongpoem/*.txt`
- 输出：每个文件生成 `*_evaluation.json`，含每题评分和跨作品平均分

## 命令

```bash
# 安装依赖（pypinyin 用于评估，bitsandbytes 需额外安装）
pip install -r requirements.txt

# 运行主程序（诗词生成）
python main.py

# 调试词表索引构建（测试 BPE tokenizer decode vs raw token）
python debug_vocab.py

# 统一评估（宋词 + 唐诗批量）
python evaluation/evaluate.py

```

所有配置集中在 `main.py:main()` 函数开头的"集中配置区域"：
- `model_name`：模型路径（支持 Qwen3-4B、Llama-3.2-3B、DeepSeek-R1-Distill-Qwen-1.5B）
- `meter_type`：`"宋词"` 或 `"唐诗"`（决定加载 `Songci_Meter/` 还是走纯算法唐诗路径）
- `rhyme_dict_name`：`"Cilin"`（词林正韵）、`"Pinshui"`（平水韵）、`"Xinyun"`（中华新韵）、`"Tongyun"`（中华通韵）
- `task_type`：`"instruction"` / `"zero-shot"` / `"one-shot"` / `"completion"`
- `use_constraints`：`True` 启用约束解码，`False` 做无约束对比
- `use_thinking`：DeepSeek R1 系列必须设为 `True`
- `num_generations`：多次生成数量
- `use_bitsandbytes`：是否启用 8-bit 量化

## 关键文件

- **Songci_Meter/\\*.json** — 宋词格律定义，每词牌一个独立 JSON 文件，支持多变体（variants 数组）
- **Rhyme/\\*.json** — 韵书（Cilin/Pinshui/Xinyun/Tongyun），韵部 → 声调 → 字列表
- **Rhyme/Cilin.json** — 词林正韵（韵部 → 声调 → 字列表）
- **Rhyme/Pinshui.json** — 平水韵
- **Rhyme/Xinyun.json** — 中华新韵
- **Rhyme/Tongyun.json** — 中华通韵
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

- `GenerationStateMachine` 和 `TangPoemStateMachine` 在每次生成开始时必须**重新初始化**，因为其内部包含韵部锁定等有状态信息
- 唐诗的重复字检测由 `tang_logits_processor.py` 专门处理，采用分层检测策略（句内重字扣分、n-gram 硬拒绝等），不再使用旧版 `-inf` 一字封杀
- BPE tokenizer（如 Qwen）的词表 token 是字节编码，需要用 `tokenizer.decode([id])` 获取真实字符，不能直接读取 token 字符串
- `[title]` 和 `[content]` 标记是程序解析生成进度的关键依赖——在遇到 `[content]` 之前约束逻辑不启动
- 唐诗 LogitsProcessor 有两级兜底：`_verifier_check()`（完整规则）→ `_critical_checks()`（仅核心格律底线）→ 最终兜底（放行全部原始分数），确保 mask 不会全 `-inf`
- `TangPoemStateMachine` 的基调确定有回退链：优先第 2 字 → 第 4 字（取反）→ 第 6 字（七言），从已生成文本中实时推断而非预设
- 评估模块（`evaluation/evaluate.py`）依赖 `pypinyin` 做平仄和韵部判断，独立于生成时使用的韵书 JSON
