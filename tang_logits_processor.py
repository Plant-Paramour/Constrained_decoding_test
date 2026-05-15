import torch
import re
from transformers import LogitsProcessor, PreTrainedTokenizer
from vocab_indexer import VocabIndexer
from tang_state_machine import TangPoemStateMachine


class TangPoemLogitsProcessor(LogitsProcessor):
    """唐诗专用 LogitsProcessor —— 集成 GLM poem_verifier.py 的格律校验逻辑

    配合 TangPoemStateMachine 和 VocabIndexer，在每步 token 采样前筛选合法候选，
    并施加二四六分明、孤平、三连同、押韵一致性、禁字过滤、重复惩罚等规则。
    """

    def __init__(self, vocab_indexer: VocabIndexer, state_machine: TangPoemStateMachine,
                 tokenizer: PreTrainedTokenizer, input_prompt_len: int):
        self.vocab_indexer = vocab_indexer
        self.state_machine = state_machine
        self.tokenizer = tokenizer
        self.input_prompt_len = input_prompt_len

        self.last_decoded_text = ""
        self.has_started_ci = False
        self.eos_token_id = tokenizer.eos_token_id

        # 全部正文文本（跨句重复检测）
        self.all_ci_text = ""

        # 标点 token 缓存
        self._init_punct_tokens()

        # token_id → 解码字符缓存
        self.token_id_to_chars = {}

        # 常见双字词集合（从 VocabIndexer 已索引 token 中提取）
        self.common_bigrams = set()
        for tid, text in self.vocab_indexer.token_to_text.items():
            if len(text) == 2:
                self.common_bigrams.add(text)

        # 句读边界位置（已生成字数到达此位置时触发跨段检测）
        if state_machine.line_length == 5:
            self._boundary_positions = {2}  # 五言 "2/3"
        else:
            self._boundary_positions = {2, 4}  # 七言 "2/2/3"

    def _init_punct_tokens(self):
        self.comma_tokens = set()
        self.period_tokens = set()
        for c in ['，', ',']:
            for tid in self.tokenizer.encode(c, add_special_tokens=False):
                self.comma_tokens.add(tid)
        for c in ['。', '.']:
            for tid in self.tokenizer.encode(c, add_special_tokens=False):
                self.period_tokens.add(tid)

    def _decode_token(self, token_id: int) -> str:
        if token_id not in self.token_id_to_chars:
            self.token_id_to_chars[token_id] = (
                self.tokenizer.decode([token_id]).replace(' ', '').replace('\r', '')
            )
        return self.token_id_to_chars[token_id]

    def _get_chars_only(self, text: str) -> str:
        return "".join(c for c in text if re.match(r'[一-龥A-Za-z]', c))

    def _get_pingze(self, char: str) -> list:
        return self.vocab_indexer.data_manager.get_pingze(char)

    def _get_rhyme_parts(self, char: str, tone: str = None) -> list:
        if tone:
            return self.vocab_indexer.data_manager.get_rhyme_part_by_tone(char, tone)
        return self.vocab_indexer.data_manager.get_rhyme_part(char)

    def _build_simulated_sheng(self, simulated_line: str) -> dict:
        """构建模拟行的字→平仄数值映射 (0=平, 1=仄)"""
        sheng = {}
        for c in simulated_line:
            pz_list = self._get_pingze(c)
            if len(pz_list) == 1:
                sheng[c] = [0] if pz_list[0] == "平" else [1]
            elif len(pz_list) > 1:
                sheng[c] = []  # 多音字放行
        return sheng

    # ── 核心：poem_verifier 风格校验 ──────────────────────────

    def _verifier_check(self, token_id: int, pos_info: dict) -> float:
        """对候选 token 施加 poem_verifier 风格格律校验。

        返回 penalty 值：<= -100 = 硬拒绝，> -100 = 从 logit 扣除的惩罚分。
        逻辑移植自 GLM poem_verifier.py 的 poem_verifier() 函数。
        """
        target_len = pos_info["target_length"]
        is_rhyming = pos_info["is_rhyming"]
        locked_rhyme_parts = pos_info.get("locked_rhyme_parts")
        rhyme_type = pos_info["rhyme_type"]
        line_tone = pos_info.get("base_tone", 2)  # 0=平起, 1=仄起, 2=未定

        # --- 正文中严禁空白与标点（检测原始解码文本，不走 strip 缓存）---
        raw_decode = self.tokenizer.decode([token_id])
        if re.search(r'[\s　，。、？！；：\n\r]', raw_decode):
            return -1000

        token_text = self._decode_token(token_id)
        token_chars = self._get_chars_only(token_text)
        if not token_chars:
            return -1000

        line_text = self.state_machine.current_line_text

        # --- 禁字过滤 (对应 verifier: 的/些/么/了) ---
        forbidden = {'的', '些', '么', '了'}
        for c in token_chars:
            if c in forbidden:
                return -1000

        # --- 模拟添加 token 后的当前行 ---
        simulated_line = line_text + token_chars
        sim_len = len(simulated_line)

        # --- 字数溢出 ---
        if sim_len > target_len:
            return -1000
        if sim_len == 6 and sim_len == target_len:
            return -1000

        # --- 确定基调（已缓存或从模拟行第2字推断）---
        if line_tone == 2 and sim_len >= 2 and len(line_text) < 2:
            pz2 = self._get_pingze(simulated_line[1])
            if len(pz2) == 1:
                line_tone = 0 if pz2[0] == "平" else 1

        # 句尾平仄期望：押韵句与韵式同调，非押韵句相反
        if is_rhyming:
            end_tone = 0 if "平" in rhyme_type else 1
        elif pos_info.get("current_line", -1) == 0:
            end_tone = 2  # 首句灵活，由实际末字决定
        else:
            end_tone = 1 if "平" in rhyme_type else 0

        # --- 二四六分明 ---
        # 第2字 (idx 1)
        if sim_len >= 2:
            pz = self._get_pingze(simulated_line[1])
            if line_tone != 2 and len(pz) == 1:
                expected = "平" if line_tone == 0 else "仄"
                if pz[0] != expected:
                    return -1000

        # 第4字 (idx 3) — 与第2字相反
        if sim_len >= 4:
            pz = self._get_pingze(simulated_line[3])
            if line_tone != 2 and len(pz) == 1:
                expected = "仄" if line_tone == 0 else "平"
                if pz[0] != expected:
                    return -1000

        # 第6字 (idx 5) — 与第2字相同（仅七言）
        if sim_len >= 6:
            pz = self._get_pingze(simulated_line[5])
            if line_tone != 2 and len(pz) == 1:
                expected = "平" if line_tone == 0 else "仄"
                if pz[0] != expected:
                    return -1000

        # --- 三连同 (行末，含多音字全组合检测) ---
        if sim_len == target_len and sim_len >= 3:
            from itertools import product
            tone_options = []
            for c in simulated_line[-3:]:
                pz_list = self._get_pingze(c)
                if not pz_list:
                    tone_options.append([None])
                else:
                    tone_options.append([0 if pz == "平" else 1 for pz in pz_list])
            for combo in product(*tone_options):
                if None in combo:
                    continue
                if sum(combo) == 0 or sum(combo) == 3:
                    return -1000

        # 注：原 poem_verifier.py:363-376 的"提前三连同"预检不适用于 token 级
        # LogitsProcessor 范式（它会检查已生成字符并可能拒绝所有候选 token）。
        # 此处改为在行完成时由三连同全量检测兜底（见上方 sim_len==target_len 处）。

        # --- 孤平 ---
        if sim_len == target_len and target_len >= 3:
            sheng_map = self._build_simulated_sheng(simulated_line)
            # 孤平仅适用于平收句（以实际末字平仄为准，而非预设 end_tone）
            last_pz_guping = sheng_map.get(simulated_line[-1], [])
            is_ping_end = len(last_pz_guping) == 1 and last_pz_guping[0] == 0
            if is_ping_end:
                # 平收的诗中：平起查 "仄平仄"@(0,1,2)，仄起查 "仄平仄"@(2,3,4)
                if line_tone == 0:
                    pz0 = sheng_map.get(simulated_line[0], [])
                    pz2_gu = sheng_map.get(simulated_line[2], [])
                    if len(pz0) == 1 and len(pz2_gu) == 1 and pz0[0] == 1 and pz2_gu[0] == 1:
                        return -1000
                elif line_tone == 1 and sim_len >= 5:
                    pz2_gu = sheng_map.get(simulated_line[2], [])
                    pz4 = sheng_map.get(simulated_line[4], [])
                    if len(pz2_gu) == 1 and len(pz4) == 1 and pz2_gu[0] == 1 and pz4[0] == 1:
                        return -1000

        # --- 句尾平仄（首句灵活，由实际末字决定收束模式）---
        if sim_len == target_len:
            last_char = simulated_line[-1]
            pz_last = self._get_pingze(last_char)
            if len(pz_last) == 1 and end_tone != 2:
                expected_tone = "平" if end_tone == 0 else "仄"
                if pz_last[0] != expected_tone:
                    return -1000

        # --- 句尾字不得与前文句尾重复（原 poem_verifier.py:287-302）---
        if sim_len == target_len and pos_info["current_line"] > 0:
            last_char = simulated_line[-1]
            for prev_line in range(pos_info["current_line"]):
                end_pos = (prev_line + 1) * target_len - 1
                if end_pos < len(self.all_ci_text) and self.all_ci_text[end_pos] == last_char:
                    return -1000

        # --- 押韵一致性（交集策略，对应原 GLM verifier）---
        if sim_len == target_len:
            # 句末字不能是"不"（原 poem_verifier.py:412-413）
            if simulated_line[-1] == '不':
                return -1000
        if sim_len == target_len and is_rhyming:
            last_char = simulated_line[-1]
            expected_tone = "平" if "平" in rhyme_type else "仄"
            rhyme_parts = set(self._get_rhyme_parts(last_char, expected_tone))
            if locked_rhyme_parts is not None and rhyme_parts:
                if not locked_rhyme_parts.intersection(rhyme_parts):
                    return -1000

        # --- 重复检测 ---
        penalty = 0.0
        for c in token_chars:
            if c in line_text:
                penalty += 3.0
            if c in self.all_ci_text:
                penalty += 1.0

        # 三字连续重复（硬拒绝，原 poem_verifier.py:254-258）
        if len(token_chars) >= 3:
            for i in range(len(token_chars) - 2):
                trigram = token_chars[i:i + 3]
                if trigram in line_text or trigram in self.all_ci_text:
                    return -1000
        # 跨边界三字重复
        if len(line_text) >= 2 and len(token_chars) >= 1:
            if (line_text[-2:] + token_chars[0]) in self.all_ci_text:
                return -1000
        if len(line_text) >= 1 and len(token_chars) >= 2:
            if (line_text[-1] + token_chars[:2]) in self.all_ci_text:
                return -1000

        # 2-gram 重复（硬拒绝）
        if len(token_chars) >= 2:
            for i in range(len(token_chars) - 1):
                bigram = token_chars[i:i + 2]
                if bigram in line_text or bigram in self.all_ci_text:
                    return -1000
        # 跨边界 2-gram
        if len(line_text) >= 1 and len(token_chars) >= 1:
            if (line_text[-1] + token_chars[0]) in self.all_ci_text:
                return -1000

        # 字必须在韵书中
        for c in token_chars:
            if not self._get_pingze(c):
                return -1000

        # --- 跨句读双字词惩罚 ---
        cur_pos = pos_info["current_char_idx"]
        if cur_pos in self._boundary_positions and line_text:
            bigram = line_text[-1] + token_chars[0]
            if bigram in self.common_bigrams:
                penalty += 8.0

        return penalty

    def _track_all_ci_text(self, latest_text: str):
        """仅维护全篇正文缓存（行文本由 state_machine 管理）"""
        for c in latest_text:
            if re.match(r'[，,。.？！；\n]', c):
                pass  # 标点不计入
            elif re.match(r'[一-龥A-Za-z]', c):
                self.all_ci_text += c

    def _handle_punctuation(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        pos_info = self.state_machine.get_position_info()
        line_idx = pos_info["current_line"]
        # 偶数句用句号，奇数句用逗号
        if (line_idx + 1) % 2 == 0:
            target_set = self.period_tokens
        else:
            target_set = self.comma_tokens
        mask = torch.full_like(scores, -float('inf'))
        for tid in target_set:
            mask[:, tid] = scores[:, tid]
        return mask

    # ── HuggingFace LogitsProcessor 接口 ──────────────────────

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 1. 解码已生成文本
        generated_ids = input_ids[0][self.input_prompt_len:].tolist()
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        raw_text = raw_text.replace(' ', '').replace('\r', '')

        # 2. 等待 [content] 标记 — 在此之前自由生成
        if not self.has_started_ci:
            if '[content]' in raw_text:
                self.has_started_ci = True
                ci_text = raw_text.split('[content]', 1)[1]
                self.state_machine.advance_state(ci_text)
                self.last_decoded_text = ci_text
                self._track_all_ci_text(ci_text)
            return scores

        # 3. 提取正文增量并更新状态
        ci_text = raw_text.split('[content]', 1)[1] if '[content]' in raw_text else raw_text
        if len(ci_text) > len(self.last_decoded_text):
            latest_chars = ci_text[len(self.last_decoded_text):]
            self.state_machine.advance_state(latest_chars)
            self.last_decoded_text = ci_text
            self._track_all_ci_text(latest_chars)

        # 4. 生成完毕 → 只允许 EOS
        if self.state_machine.is_finished:
            mask = torch.full_like(scores, -float('inf'))
            mask[:, self.eos_token_id] = scores[:, self.eos_token_id]
            return mask

        # 5. 需要标点 → 输出标点
        if self.state_machine.needs_punctuation:
            return self._handle_punctuation(scores)

        # 6. 获取合法模式并收集候选 token
        allowed_patterns = self.state_machine.get_allowed_patterns()
        allowed_tokens = set()
        for length, pz_pattern, rhyme_req in allowed_patterns:
            base_set = self.vocab_indexer.pattern_tokens.get((length, pz_pattern), set())
            if rhyme_req == "ANY_RHYME":
                rhyme_type = pz_pattern[-1]
                valid_rhyme = set()
                for (rt, rp), t_ids in self.vocab_indexer.rhyme_tokens.items():
                    if rt == rhyme_type:
                        valid_rhyme.update(t_ids)
                if valid_rhyme:
                    base_set = base_set.intersection(valid_rhyme)
            elif rhyme_req is not None:
                rhyme_type = pz_pattern[-1]
                valid_rhyme = self.vocab_indexer.rhyme_tokens.get((rhyme_type, rhyme_req), set())
                if valid_rhyme:
                    base_set = base_set.intersection(valid_rhyme)
            allowed_tokens.update(base_set)

        # 7. 构建掩码并施加 verifier 规则
        mask = torch.full_like(scores, -float('inf'))
        vocab_size = scores.shape[1]

        # 过滤越界 token（vocab_indexer 的 token ID 可能超出模型实际 vocab 范围）
        allowed_list = [tid for tid in allowed_tokens if 0 <= tid < vocab_size]
        if not allowed_list:
            mask[:, self.eos_token_id] = scores[:, self.eos_token_id]
            return mask

        allowed_tensor = torch.tensor(allowed_list, dtype=torch.long, device=scores.device)
        token_scores = scores[0, allowed_tensor].clone()
        pos_info = self.state_machine.get_position_info()

        for idx, t_id in enumerate(allowed_list):
            penalty = self._verifier_check(t_id, pos_info)
            if penalty <= -100:
                token_scores[idx] = -float('inf')
            else:
                token_scores[idx] -= penalty

        # 所有候选均被硬拒绝 → EOS 兜底，避免 CUDA 断言
        if (token_scores == -float('inf')).all():
            mask[:, self.eos_token_id] = scores[:, self.eos_token_id]
            return mask

        mask[0, allowed_tensor] = token_scores
        return mask
