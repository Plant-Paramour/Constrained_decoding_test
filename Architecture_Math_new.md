# 适用于自回归模型的通用诗歌生成框架------基于格律规则的约束解码

## 1. 生成状态自动机 (Generation State Machine)

### 本质是一个确定性有限状态机（Deterministic Finite-State Machine, FSM），用于在自回归生成的每一步 𝑡 追踪当前结构约束，并决定下一步的合法生成动作。

### 1.1 状态空间定义

模型在第 $t$ 步的生成状态定义为一个多元组 $S_{t}$：

$$S_{t} = \{\pi,k,c,N_{punct},N_{newline},N_{caesura},E_{rhyme}\}$$

$\pi$：当前所在的诗体或词牌（Stanzas）。

$k$：当前正在生成的行索引。

$c$：当前行已生成的字符数（$L_{k}$ 为当前行目标字符总长）。
$N_{punct},N_{newline},N_{caesura} \in \{ 0,1\}$：布尔控制位，分别表示当前状态是否必须生成标点、换行或顿号。

$E_{rhyme}$：当前已锁定的局部韵部集合字典。

不涉及状态空间搜索或路径规划，而是用于**约束驱动的自回归生成**，不替代或描述大语言模型的内部状态。

### 1.2 合法模式推导函数 (State-to-Pattern Mapping)

在非标点状态下，我们需要求解下一步允许的所有（长度，平仄，押韵）组合集合
$A_{t}$。 令当前行的纯格律字符串为
$P_{k}$（去除了句读符`“/”`），剩余可生成的字数为
$R = L_{k} - c$。令距离最近的句读位剩余字数为 $R_{break}$。

对任意候选长度
$l \in \lbrack 1,min(l_{\max},R_{break})\rbrack$，提取目标格律子串。定义平仄展开算子
$\Phi$ ：

$$\Phi(P_{k}\lbrack c:c + l\rbrack) = \left\{ p \mid p\text{ }\text{将}\text{所有的 ’中’ 展开为 ’平’ 或 ’仄’} \right\}$$

当且仅当 $l = R$ 且当前段句定义为押韵句时，激活押韵约束条件集
$V_{rhyme}$；否则为 $None$。 状态机向外输出的全合法模式集为：

$$ A_t = \left\{ (l, p, r) \mid 1 \le l \le \min(l_{max}, R_{break}), \; p \in \Phi(P_k[c : c+l]), \; r \in V_{rhyme} \right\} $$

## 2. Logits 干预器 (Constraint Logits Processor)

干预器是一个挂载于模型输出流上的概率掩码与惩罚函数。令大语言模型在第 $t$
步前向传播输出的原始对数几率向量为 $Z \in \mathbb{R}^{|V|}$，其中 $|V|$
为分词器词表大小。需要计算修正后的向量 $Z'$。

### 2.1 结构性硬约束截断 (Structural Hard Masking)

对于标点、换行、断句掩码控制： 令 $V_{punct}$ 为当前要求下合法的标点符号
Token 集合。若对应的控制位（如 $N_{punct}$）为 $1$，则：

$$Z_{i}' = \left\{ \begin{matrix}
Z_{i} & \text{if }i \in V_{punct} \\
 - \infty & \text{otherwise}
\end{matrix} \right.\ $$

### 2.2 词表合法空间投影 (Valid Token Space Projection)

当进入正文生成阶段时，干预器将状态机输出的模式集 $A_{t}$
映射到当前可用的 Token 白名单 $V_{valid}$ 集合上。

$$V_{valid} = \bigcup_{(l,p,r) \in A_{t}}\left( T_{len,tone}(l,p) \cap T_{rhyme}(r) \right)$$

($T_{len,tone}$ 为通过预计算词表查到的"长度+平仄"合规 Token
集合，$T_{rhyme}$ 为符合该韵部约束的 Token 集合。)

### 2.3 动态时间距离重复惩罚 (Exponential Decay Repetition Penalty)

为防止强制受限解码带来的"贪婪重复"，引入基于字符绝对出现位置的指数级重复惩罚项
$\rho$。

对任意处于 $V_{valid}$ 中的 Token 对应的词内所含有的汉字字符
$c$，若在先前生成的全正文中出现过，且上一次出现的绝对位置为
$pos(c)$，当前全局绝对位置为 $P_{cur}$。定义该字符的时空距离
$d = max(1,P_{cur} - pos(c))$。

惩罚项算符（宋词模式）：

$$\rho(c) = \left\{ \begin{matrix}
\max\left( 8.0,20.0 \cdot e^{- 0.05d} \right) & if\ c \in already\ used\ characters \\
0 & \text{otherwise}
\end{matrix} \right.\ $$

极端惩罚项算符（唐诗模式）：

$$\rho(c) = \left\{ \begin{matrix}
 + \infty & if\ \ c \in already\ used\ characters \\
0 & \text{otherwise}
\end{matrix} \right.\ $$

对于一个包含多个字符的 Token $w_{i}$，其累计惩罚总值
$Pen(w_{i}) = \sum_{c \in w_{i}}^{}\rho(c)$。

### 2.4 后验 Logits 生成方程 (Final Logits Update)

干预器最终向模型返回的修改后 Logits 值为：

$$Z_{i}' = \left\{ \begin{matrix}
Z_{i} - Pen(w_{i}) & \text{if }i \in V_{valid} \\
 - \infty & \text{otherwise}
\end{matrix} \right.\ $$

经由底层的 Softmax 激活函数，在数学定义上确保了不在 $V_{valid}$ 中的
Token 生成概率严格收敛至 $0$。
