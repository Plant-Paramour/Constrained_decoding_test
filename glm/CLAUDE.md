# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 GLM-10B-Chinese 模型的中国古典诗词（五言/七言绝句和律诗）自动生成系统。核心依赖**受限束搜索（Constrained Beam Search）**与**物理格律规则过滤**两大模块。

## 运行命令

```bash
# 批量生成（带迭代优化）
bash run_bipro_poem.sh config/glm_10b_chinese.sh 0

# 单次生成（无优化，更快）
bash run_direct_poem.sh config/glm_10b_chinese.sh 0

# 直接运行 Python
python bipro_poem.py --mode inference --model-parallel-size 1 --num-beams 6 --fp16 --batch-size 6 --device 0
```

运行前需修改 `config/glm_10b_chinese.sh` 中的 `WEIGHT_PATH` 为实际的模型权重路径。

## 核心架构

### 两套生成流程

| 脚本 | 策略类 | 特点 |
|---|---|---|
| `bipro_poem.py` | `MultiGenStrategy` (mtgen.py) | 逐句生成 + 逆提示重排序 + 全诗迭代优化(refine) |
| `inference_poems_singlegen.py` | `SingleGenStrategy` (single_gen.py) | 逐句直接生成，无优化，速度快 |

两套流程共享相同的模板引擎：输入格式为 `诗歌《标题》 作者:李白 体裁:格律诗 标题:xxx 正文:[gMASK]`，通过 `[gMASK]`/`[sMASK]` 占位符控制逐句自回归生成。

### 生成策略类（采样层）

所有策略类实现相同的接口 `forward(logits, tokens, mems)` → `finalize()`，在每一步 token 采样时实时调用 `poem_verifier()` 过滤违规候选。

- `BeamSearchStrategy` (beam_search_strategy.py): 经典束搜索，带 verifier 回调
- `MultiGenStrategy` (mtgen.py): 多样本采样策略，无束搜索重排序
- `SingleGenStrategy` (single_gen.py): 简单多样本采样
- `iPromptStrategy` (iprompt.py): 集成逆提示评分评估的完整策略

### 格律校验器 (poem_verifier.py)

`poem_verifier(sentence, verifier_params)` — 核心过滤函数，被策略类在每一步 token 采样时调用。返回 `-1000` 表示违规（丢弃该候选），返回 `>100` 表示完整句通过。

校验规则：
- **字数限制**：强制 5 或 7 字，禁止 6 字
- **禁字过滤**：禁止"的"、"些"、"么"、"了"等现代白话虚词
- **重字查重**：防止句内和相邻句间重复
- **平仄卡位**：严格遵循"二四六分明"，检查第 2/4/6 字平仄
- **孤平检测**：特定句式下检测孤平
- **句尾三连同**：禁止三连平/三连仄
- **押韵一致性**：基于《平水韵》(pingshui.txt) 一韵到底
- **多音字妥协**：多音字（`len(pz)>1`）直接放行

辅助函数：`pingshui()` 加载平水韵词典，`verify_rhy()` 确定全诗韵脚和首句平仄基调。

### 质量评分 (ip_score.py / ppl.py)

- `compute_ip()`: 逆提示评分 — 给定生成的诗句，反向预测标题、情感、上下文句的困惑度，越低越好
- `aggregate_scores()`: 加权聚合标题/情感/上下文三个维度的分数
- `get_correspond_sentence()`: 从生成结果中提取 [sMASK] 对应的前句或后句
- `ppl()` (ppl.py): 底层困惑度计算，支持指定 `log_attention_weights` 加权

在 `bipro_poem.py` 中，每个 beam 候选通过 `compute_ip` 评分重排序；8句版和4句版比较后优选。

### 数据流

1. 读入 `titles2.txt`（每行一个标题，可选 `|` 指定首字、空格指定情感）
2. `process()` → `poemgen()` 逐句生成（交替 [gMASK]/[sMASK]）
3. 每句生成后调用 `verify_rhy()` 更新韵部/平仄状态
4. `bipro_poem.py` 额外执行 `refine_poem()` 对每句进行迭代优化
5. 输出保存到 `generated_poems_iprompt/` 或 `generated_single/`

### 关键依赖

- `sat` (SwissArmyTransformer): GLM 模型框架，提供 `GLMModel`, `get_tokenizer`, `get_args` 等
- `pynvml`: GPU 内存监控
- 词典文件: `pingshui.txt`（平水韵）、`cilin.txt`（词林，cilin() 函数加载但主要流程用 pingshui）

### 格式识别

`judge_type()` 函数从 `human.txt` 中自动识别诗歌格式（五绝/五律/七绝/七律），支持的格式列表见 `supported_list.txt`。
