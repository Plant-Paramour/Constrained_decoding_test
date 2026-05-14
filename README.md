# Constrained Decoding Test (古典诗词受限解码生成)

本项目是一个基于大型语言模型（LLM）的**古典诗词受限解码流水线**。它通过干预模型在生成阶段的 Logits，使大型语言模型（如 Qwen, Llama3 等）能够直接生成严格符合格律（平仄、押韵、字数、句式标点）的宋词或唐诗，无需后处理或反复重试。

## 📁 目录结构及文件介绍

| 文件/目录 | 作用说明 |
| --------- | -------- |
| `main.py` | **程序入口与编排中心**。<br>负责加载 LLM 及 Tokenizer，配置模型参数（如 8-bit 量化）；构建各种模式（Zero-shot, One-shot, Constrained 等）的 Prompt 模板；最后将状态机和干预器挂载到模型的 `generate` 函数中执行生成并保存结果。 |
| `data_manager.py` | **数据中心（Data Manager）**。<br>负责读取 `Meter/`（格律模版）和 `Rhyme/`（韵书字典），在内存中构建“字 -> 韵部/平仄”的映射字典（如 `char_to_rhyme_tone`），并提供快捷的数据查询接口供其它组件使用。 |
| `state_machine.py` | **生成状态机（State Machine）**。<br>项目的规则引擎神经中枢。根据当前选择的词牌/诗体格律，维护模型当前的生成进度（第几段、第几句、已经输出了多少字）。计算下一步允许输出的平仄规律、是否需要断句/打标点、以及句末用哪个韵部。 |
| `logits_processor.py` | **Logits 干预器（Logits Processor）**。<br>继承自 HuggingFace 的 `LogitsProcessor`。它会在 LLM 每生成一个字符预测下一个词分布（Logits）的瞬间拦截生成过程。通过查询状态机，强制把不符合平仄、押韵要求或在不该打标点时打标点的 token 概率直接变为负无穷（`-inf`），从而迫使模型只能选择合规的字。 |
| `vocab_indexer.py` | **词表属性索引器（Vocab Indexer）**。<br>用于解决查询性能问题。在加载模型后，扫描 Tokenizer 的整个词表，剔除无用符号，为所有中文 Token 提前计算好长度、平仄组合（单字或多字词组），以及是否押韵。这些将被用来在 `logits_processor` 中极速过滤候选合法 Token。 |
| `debug_vocab.py` | 辅助调试脚本。用于开发者排查词表 Token 切分、乱码或特殊字的平仄韵部归属问题。 |
| `Meter/` | 收录诗词格律模板配置的目录，包含 `songci.json`（宋词格律，含词牌）和 `TongPoem.json`（唐诗格律，含七律、五绝等）。 |
| `Rhyme/` | 韵书数据存储目录。包含 `Pinshui.json`（平水韵, 唐诗主要用）、`Cilin.json`（词林正韵, 宋词主要用）和 `Xinyun.json`（中华新韵）。 |
| `output/` | 模型生成的诗词文本结果输出目录，按词牌/诗体分类保存。 |

---

## ⚙️ 核心流程与函数解析

### 1. 词表预处理与特征绑定 (`vocab_indexer.py`)
在模型推理前，程序会通过 `VocabIndexer._build_index()` 函数遍历 Tokenizer 的整个词库构建反向索引：
- **`pattern_tokens`**：记录 `(截断长度, 平仄排列) -> 可用 Token ID 集合` 的映射。例如查询 `(2, "平仄")` 对应的键，就可以一口气拿出所有词库中发音为“平仄”的二字词语 token。
- **为什么要映射多字词？** 现代 LLM 的 Tokenizer 经常将两个汉字（如“不知”）合并为一个 Token。该索引使受限解码器也能处理多汉字的联合 Token 生成。

### 2. 状态机推演规划 (`state_machine.py`)
在每次生成开始前，都会实例化 `GenerationStateMachine`，核心包括：
- **`get_allowed_patterns()`**：核心函数。读取当前句子还剩多少字，结合当前句格律（如 `"中中/平平/中仄平"`），枚举接下来可以生成的全合法组合。例如剩余 3 个字且结尾须押“东”韵，它会返回允许的长度（L=1~3），强制该长度结尾字符合特定的押韵和固定平仄条件（同时会解释模糊匹配的“中”字）。
- **`advance_state(text: str)`**：一旦模型真正生成了字符，通过此函数通知状态机“向前走”，它会自动进位，决定是否该换行或请求触发下一个标点符号规则（`needs_punctuation` / `needs_newline`）。

### 3. Logits 的硬性截断裁剪 (`logits_processor.py`)
模型推理主干通过调用 `ConstraintLogitsProcessor.__call__` 来应用上述规则：
- **状态同步**：从模型的原始文本解析出目前为止生成的内容，剥离 `[content]` 等 Prompt 前缀，然后抛给 `state_machine.advance_state()` 定位模型走到哪了。
- **打标点强制**：如果状态机报告 `needs_punctuation` (必须打标点)，LogitsProcessor 会把满词汇表概率设为 `-inf`，唯独开放对应的奇/偶句标点（如逗号、句号），即逼迫模型必须在这回合输出特定标点。
- **合法截断过滤**：对于正文字符，调用 `vocab_indexer` 取出所有的合法白名单 token 集合。所有不在合法列表的对应概率全被压平到 `-inf`。同时为了对多字生搬硬凑有所控制，其内置了一套对诗词内已生成中文字符的动态指数级**重复字惩罚机制**（惩罚近期重复率以防止车轱辘话）。

### 4. 从 Prompt 到结果 (`main.py`)
- **`build_prompt_messages` / `build_tangpoem_prompt_messages`**：构造包含 Role-play 与特定格式输出要求（`[title]` 和 `[content]` 限定符极其严格）的 Instruction Prompt。
- 将以上的 LogitsProcessor 打包进 `LogitsProcessorList`，传入 `model.generate(...)` 中。由底层的 HuggingFace CausalLM 执行逐字自回归。因为每走到下一步都会被 LogitsProcessor 限定范围，最终的输出一定是 100% 遵守格律配置和平仄韵书的。

## 💡 总结说明

该项目是一种从底层干涉模型生成的算法实现。它的独特优势是**生成效率极高且保证格式 0 错误率**，比单纯依赖 Prompt（哪怕是 Few-shot 或强化微调）让大模型自主“理解”字数、平仄与押韵要稳定得多。开发者可通过修改 `Meter/` 下的 JSON 以及切换 `main.py` 的模型或参数，立刻将同一套机制运用在新微调或其它强大的语言模型上。
