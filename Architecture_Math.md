# 适用于自回归模型的通用诗歌生成框架——基于格律规则的约束解码

## 1. 状态规划自动机 (Generation State Machine)
状态规划机本质是一个确定性有限自动机（Deterministic Finite Automaton, DFA），用于在自回归生成的每一步 $t$ 追踪和推导合规的约束路径。

### 1.1 状态空间定义
模型在第 $t$ 步的生成状态定义为一个多元组 $S_t$：
$$S_t = \{ \pi, k, c, N_{punct}, N_{newline}, N_{caesura}, E_{rhyme} \}$$
*   $\pi$：当前所在的诗体宏组合或词牌（Stanzas）。
*   $k$：当前正在生成的行索引。
*   $c$：当前行已生成的字符数（$L_k$ 为当前行目标字符总长）。
*   $N_{punct}, N_{newline}, N_{caesura} \in \{0, 1\}$：布尔控制位，分别表示当前状态是否必须生成标点、换行或顿号。
*   $E_{rhyme}$：当前已锁定的局部韵部集合字典。

### 1.2 合法模式推导函数 (State-to-Pattern Mapping)
在非标点状态下，我们需要求解下一步允许的所有（长度，平仄，押韵）组合集合 $A_t$。
令当前行的纯格律字符串为 $P_k$（去除了分词符 `/`），剩余可生成的字数为 $R = L_k - c$。令距离最近的局部断句点或句末的字数为 $R_{break}$。

对任意候选长度 $l \in [1, \min(l_{max}, R_{break})]$，提取目标格律子串。定义平仄展开算子 $\Phi$ (对应代码 `_expand_zhong`)：
$$ \Phi(P_k[c : c+l]) = \left\{ p \mid p \text{ 展开所有的 '中' 为 '平' 或 '仄'} \right\} $$

当且仅当 $l = R$ 且当前段句定义为押韵句时，激活押韵约束条件集 $V_{rhyme}$；否则为 $None$。
状态机向外输出的全合法模式集为：
$$ A_t = \left\{ (l, p, r) \mid 1 \le l \le \min(l_{max}, R_{break}), \; p \in \Phi(P_k[c : c+l]), \; r \in V_{rhyme} \right\} $$

---

## 2. Logits 干预器 (Constraint Logits Processor)
干预器是一个挂载于模型输出流上的概率掩码与惩罚函数。令大语言模型在第 $t$ 步前向传播输出的原始对数几率向量为 $Z \in \mathbb{R}^{|V|}$，其中 $|V|$ 为分词器词表大小。需要计算修正后的向量 $Z'$。

### 2.1 结构性硬约束截断 (Structural Hard Masking)
对于标点、换行、断句掩码控制：
令 $V_{punct}$ 为当前要求下合法的标点符号 Token 集合。若对应的控制位（如 $N_{punct}$）为 $1$，则：
$$ Z'_i = \begin{cases} Z_i & \text{if } i \in V_{punct} \\ -\infty & \text{otherwise} \end{cases} $$

### 2.2 词表合法空间投影 (Valid Token Space Projection)
当进入正文生成阶段时，干预器将状态机输出的模式集 $A_t$ 映射到当前可用的 Token 白名单 $V_{valid}$ 集合上。
$$ V_{valid} = \bigcup_{(l, p, r) \in A_t} \Big( T_{len, tone}(l, p) \cap T_{rhyme}(r) \Big) $$
*($T_{len, tone}$ 为通过预计算词表查到的“长度+平仄”合规 Token 集合，$T_{rhyme}$ 为符合该韵部约束的 Token 集合。)*

### 2.3 动态时间距离重复惩罚 (Exponential Decay Repetition Penalty)
为防止强制受限解码带来的“贪婪重复”，引入基于字符绝对出现位置的指数级重复惩罚项 $\rho$。

对任意处于 $V_{valid}$ 中的 Token 对应的词内所含有的汉字字符 $c$，若在先前生成的全正文中出现过，且上一次出现的绝对位置为 $pos(c)$，当前全局绝对位置为 $P_{cur}$。定义该字符的时空距离 $d = \max(1, P_{cur} - pos(c))$。

惩罚项算符（非唐诗模式）：
$$ \rho(c) = \begin{cases} \max\left( 8.0, 20.0 \cdot e^{-0.05 d} \right) & c \in \text{已出现字符} \\ 0 & \text{otherwise} \end{cases} $$

极端惩罚项算符（唐诗模式 `is_tangpoem_flag = True`）：
$$ \rho(c) = \begin{cases} +\infty & c \in \text{已出现字符} \\ 0 & \text{otherwise} \end{cases} $$

对于一个包含多个字符的 Token $w_i$，其累计惩罚总值 $Pen(w_i) = \sum_{c \in w_i} \rho(c)$。

### 2.4 后验 Logits 生成方程 (Final Logits Update)
干预器最终向模型返回的修改后 Logits 值为：
$$ Z'_i = \begin{cases} Z_i - Pen(w_i) & \text{if } i \in V_{valid} \\ -\infty & \text{otherwise} \end{cases} $$
经由底层的 Softmax 激活函数，在数学定义上确保了不在 $V_{valid}$ 中的 Token 生成概率严格收敛至 $0$。


