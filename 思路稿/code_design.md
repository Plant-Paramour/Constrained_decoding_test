# 格律约束解码：代码实现思路与架构设计

在进入具体代码编写前，我们需要将核心架构进行模块化拆分。根据“贪心查表+状态机约束”的思路，代码设计可以分为以下几个核心模块：

## 1. 核心类与模块设计

### 1.1 数据管理模块 (`DataManager`)
- **职责**：负责加载和解析基础规则数据。
- **核心方法**：
  - `load_cilin(filepath)`: 解析《词林正韵》，构建双向映射：`字 -> [(平仄, 韵部)]` 以及 `韵部 -> [字]`。注意处理多音字不同平仄的情况。
  - `load_cipai(filepath, name)`: 加载特定词牌（如“浣溪沙”）的格律规则，解析出结构化数据（几片、每片几句、句读位置、平仄要求及押韵位置）。

### 1.2 词表预处理与索引构建 (`VocabIndexer`)
- **职责**：在模型加载后“离线”运行一次，遍历所有的 Token，提前判断它们的格律属性，空间换时间，极大提升推理速度。
- **存储结构**：
  - `token_to_text`: `Dict[int, str]`，TokenID 到纯中文文本的映射（非中文、标点自动剔除）。
  - `pattern_tokens`: `Dict[Tuple[int, str], Set[int]]`，映射 `(长度, 平仄模式) -> 合规TokenID集合`。
    - *例如*：`(2, "平仄") -> {ID_A, ID_B...}`。其中多音字Token会命中多个模式。
  - `rhyme_tokens`: `Dict[Tuple[str, str], Set[int]]`，针对专门的押韵位置，建立 `(韵脚类别, 韵部) -> TokenID集合`。

### 1.3 生成状态机 (`GenerationStateMachine`)
- **职责**：在逐步生成中，负责追踪和推进格律匹配进度。
- **内部状态**：
  - `current_stanza`, `current_line`, `current_segment`: 定位当前正在生成哪阕、哪句、哪个句读。
  - `segment_remains`: 当前句读（由`/`划分的片段）还差多少个字填满。
  - `target_pattern`: 当前片段剩余字所要求的平仄（如 `"平平仄"`）。
  - `locked_rhyme`: 第一处押韵确立的韵部，若未建立则为 `None`。
- **核心方法**：
  - `get_allowed_patterns()`: 探查当前状态允许哪些 `(长度, 平仄, [可选:韵部])` 的组合。
  - `advance_state(token_text)`: 接收新生成的字符串，减少 `segment_remains`，后移指针。如果跨句读或跨句，自动跳转到下一阶段。如果是首个押韵位，则更新 `locked_rhyme`。

### 1.4 逻辑运算与掩码注入 (`ConstraintLogitsProcessor`)
- **职责**：直接与推断引擎（如基于 HuggingFace transformers，考虑到 Windows 环境，暂不使用 vLLM）对接，在模型输出原始 logits 后、进行 softmax 概率计算和采样前，对 logits 进行 masking。
- **核心方法**：
  - `__call__(input_ids, scores)`: 
    1. 通过 `input_ids` 的变化识别出上一轮刚生成的 token。
    2. 调用状态机的 `advance_state()` 更新状态。
    3. 调用状态机的 `get_allowed_patterns()` 明确当前允哪些格律模式。
    4. 从 `VocabIndexer` 获取这些模式对应的合规 TokenID 子集（取并集）。
    5. 创建全 `-inf` 的掩码张量，仅保留合规 TokenID 对应的原始分数。
    6. 将过滤后的 logits 返回给框架去执行采样。

---

## 2. 关键难点与处理策略

### 2.1 “中”字诀（可平可仄）的处理
格律中的“中”代表平仄均可。在状态机请求 `get_allowed_patterns` 时：
- 如果当前剩余序列是 `"中仄"`，状态机会展开为两组允许的模式：`"平仄"` 和 `"仄仄"`。
- 将这两组模式分别去 `VocabIndexer` 查询 Token ID，最后求并集，保证多字 Token 的匹配。

### 2.2 句读硬阻断（长度约束）
为确保类似“一只/小狗”被截断成2字，若当前断句剩余字数为 $N$，我们只从 Token 中索求长度 $L \le N$ 的 字，彻底杜绝了模型一次性吐出 $N+1$ 个字的 Token。

### 2.3 状态同步延迟风险
在 HuggingFace 的 `LogitsProcessor` 工作机制中，输入的是当前已有的 `input_ids`，输出必须是针对**下一个词**预测的 `logits`。我们需要确保：`当前光标位置 = input_ids 解码后的汉字总数 - 初始Prompt字数`，基于这个绝对位置来同步状态机，比相对计算更稳健。

### 2.4 补充强调
模型只能输出汉字和标点，任何非汉字的 Token 都不应被允许。`VocabIndexer` 在预处理阶段就会剔除掉所有非汉字的 Token，确保后续逻辑中不需要再进行额外检查。

### 2.5 模型通用性与适配
最终代码必须具备高度的可复用性，能够独立于模型具体架构运行。当前首要目标是确保构建的干预逻辑在 **Qwen3** 模型上 **100% 能够运行且可用**。同时，由于底层基于 HuggingFace transformers 框架，设计上需保证未来能够直接复用到如 DeepSeek 等其他开流大语言模型上，做到“核心约束算法一套通用”，仅根据不同模型调整 Model 和 Tokenizer 的实例化代码。

## 3. 执行流程图示

```text
[加载数据与模型]
      ↓
[离线预处理: VocabIndexer 扫描词表分类, 耗时约数秒到数分钟]
      ↓
[初始化目标词牌状态机] ——————> (初始状态：浣溪沙, 剩余字数:2)
      ↓
[循环生成下一个Token]
  ├─ 1. 模型前向传播产生 Logits [vocab_size]
  ├─ 2. LogitsProcessor 获取允许的 Token IDs 集合
  ├─ 3. 将非法位置写死为 -inf
  ├─ 4. 模型从合法范围中采样出 Token '一' (长度1)
  └─ 5. 解析 Token，状态机更新 (剩余字数:2 -> 1)
      ↓
[直至全词跑完约束结束]
```

