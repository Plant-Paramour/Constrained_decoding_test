import torch
import math
import re
from transformers import LogitsProcessor, PreTrainedTokenizer
from vocab_indexer import VocabIndexer
from tang_state_machine import TangPoemStateMachine


class TangPoemLogitsProcessor(LogitsProcessor):
    """唐诗专用 LogitsProcessor —— 集成 GLM poem_verifier.py 的格律校验逻辑

    在每步生成时，根据状态机的允许平仄模式配合 vocab_indexer 筛选合法 token，
    并额外施加唐诗特有的格律规则：二四六分明、孤平、三连同、押韵一致性、禁字过滤、重复惩罚。
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

        # 当前句已生成的字面文本（用于位置平仄校验、重复检测）
        self.current_line_text = ""
        # 全部正文文本（跨句重复检测）
        self.all_ci_text = ""
        # 已押韵的字列表（用于押韵一致性校验，对应原 verifier 的 yayun）
        self.rhyme_chars = []

        # 标点 token 缓存
        self._init_punct_tokens()

        # token_id → 解码字符缓存
        self.token_id_to_chars = {}

        # 缓存当前行的位置平仄期望（在每句开始时计算）
        self._cached_line_tone = 2  # 0=平起, 1=仄起, 2=未定

    def _init_punct_tokens(self):
        """初始化逗号和句号对应的 token ID 集合"""
        self.comma_tokens = set()
        self.period_tokens = set()

        comma_chars = ['，', ',']
        period_chars = ['。', '.']

        for c in comma_chars:
            ids = self.tokenizer.encode(c, add_special_tokens=False)
            for tid in ids:
                self.comma_tokens.add(tid)

        for c in period_chars:
            ids = self.tokenizer.encode(c, add_special_tokens=False)
            for tid in ids:
                self.period_tokens.add(tid)

    def _decode_token(self, token_id: int) -> str:
        if token_id not in self.token_id_to_chars:
            clean_text = self.tokenizer.decode([token_id]).replace(' ', '').replace('\r', '')
            self.token_id_to_chars[token_id] = clean_text
        return self.token_id_to_chars[token_id]

    def _get_chars_only(self, text: str) -> str:
        return "".join(c for c in text if re.match(r'[一-龥A-Za-z]', c))

    def _get_pingze(self, char: str) -> list:
        """获取字的平仄列表，返回 [] 表示不在韵书中"""
        return self.vocab_indexer.data_manager.get_pingze(char)

    def _get_rhyme_parts(self, char: str, tone: str = None) -> list:
        if tone:
            return self.vocab_indexer.data_manager.get_rhyme_part_by_tone(char, tone)
        return self.vocab_indexer.data_manager.get_rhyme_part(char)

    def _get_line_tone(self, pos_info: dict) -> int:
        """获取当前诗句的基调：0=平起, 1=仄起, 2=未定"""
        line_text = self.current_line_text
        if len(line_text) < 2:
            return 2

        # 从第2字确定基调（原 verifier 的逻辑）
        pz = self._get_pingze(line_text[1])
        if len(pz) == 1:
            return 0 if pz[0] == "平" else 1
        return 2

    def _get_end_tone(self, pos_info: dict) -> int:
        """获取当前句尾期望的平仄：0=平收, 1=仄收"""
        line_idx = pos_info["current_line"]
        # 奇句仄收，偶句平收（平韵）；反之（仄韵）
        if "平" in pos_info["rhyme_type"]:
            return 0 if (line_idx + 1) % 2 == 0 else 1
        else:
            return 1 if (line_idx + 1) % 2 == 0 else 0

    def _build_simulated_sheng(self, simulated_line: str) -> dict:
        """构建模拟行的字→平仄数值映射 (0=平, 1=仄)"""
        sheng = {}
        for c in simulated_line:
            pz_list = self._get_pingze(c)
            if len(pz_list) == 1:
                sheng[c] = [0] if pz_list[0] == "平" else [1]
            elif len(pz_list) > 1:
                sheng[c] = []  # 多音字，标记为空列表（不参与严格检查）
            # 不在韵书中的字不加入
        return sheng

    def _verifier_check(self, token_id: int, pos_info: dict) -> float:
        """
        对候选 token 施加 poem_verifier 风格的格律校验。
        返回 penalty 值：<= -100 表示硬拒绝，> -100 表示惩罚分（越大越好）。
        这是对原 GLM poem_verifier() 核心逻辑的适配移植。
        """
        char_idx = pos_info["current_char_idx"]
        target_len = pos_info["target_length"]
        is_rhyming = pos_info["is_rhyming"]
        locked_rhyme = pos_info["locked_rhyme_part"]
        rhyme_type = pos_info["rhyme_type"]

        token_text = self._decode_token(token_id)
        token_chars = self._get_chars_only(token_text)
        if not token_chars:
            return -1000

        # --- 禁字过滤 (对应 verifier 中 的/些/么/了 检查) ---
        forbidden = {'的', '些', '么', '了'}
        for c in token_chars:
            if c in forbidden:
                return -1000

        # --- 模拟添加 token 后的当前行 ---
        simulated_line = self.current_line_text + token_chars
        sim_len = len(simulated_line)

        # --- 字数溢出检查 ---
        if sim_len > target_len:
            return -1000

        # --- 6字禁 (原 verifier: 6字绝句不存在) ---
        if target_len == 5 and sim_len == target_len:
            pass  # 五言正常
        if sim_len == 6 and sim_len == target_len:
            return -1000  # 不存在六言句
        # 但实际上 max_length 由格律决定，不会出现 6 字目标

        # --- 二四六分明：位置平仄卡位 ---
        # 确定当前行的平仄基调（优先从已生成文本，不足时才从模拟行推断）
        line_tone = self._cached_line_tone
        if line_tone == 2 and sim_len >= 2 and len(self.current_line_text) < 2:
            pz2 = self._get_pingze(simulated_line[1])
            if len(pz2) == 1:
                line_tone = 0 if pz2[0] == "平" else 1

        end_tone = self._get_end_tone(pos_info)

        # 第2字 (index 1) 检查
        if sim_len >= 2:
            pz = self._get_pingze(simulated_line[1])
            if line_tone != 2 and len(pz) == 1:
                expected = "平" if line_tone == 0 else "仄"
                if pz[0] != expected:
                    return -1000

        # 第4字 (index 3) 检查 — 必须与第2字相反
        if sim_len >= 4:
            pz = self._get_pingze(simulated_line[3])
            if line_tone != 2 and len(pz) == 1:
                expected = "仄" if line_tone == 0 else "平"
                if pz[0] != expected:
                    return -1000

        # 第6字 (index 5) 检查 — 必须与第2字相同
        if sim_len >= 6:
            pz = self._get_pingze(simulated_line[5])
            if line_tone != 2 and len(pz) == 1:
                expected = "平" if line_tone == 0 else "仄"
                if pz[0] != expected:
                    return -1000

        # --- 三连同预防 (行末倒数第3-2-1字) ---
        if sim_len == target_len and sim_len >= 3:
            sheng_map = self._build_simulated_sheng(simulated_line)
            last3 = [sheng_map.get(c, []) for c in simulated_line[-3:]]
            if all(len(p) == 1 for p in last3):
                tones = [p[0] for p in last3]
                if sum(tones) == 0:  # 三连平
                    return -1000
                if sum(tones) == 3:  # 三连仄
                    return -1000

        # --- 孤平检测 (行将完成时) ---
        if sim_len == target_len and target_len >= 3:
            sheng_map = self._build_simulated_sheng(simulated_line)
            if end_tone == 0:  # 平收
                if line_tone == 0:  # 平起 → 检查 "仄平仄" 在位置 0,1,2
                    pz0 = sheng_map.get(simulated_line[0], [])
                    pz2_gu = sheng_map.get(simulated_line[2], [])
                    if len(pz0) == 1 and len(pz2_gu) == 1:
                        if pz0[0] == 1 and pz2_gu[0] == 1:
                            return -1000
                elif line_tone == 1:  # 仄起 → 检查 "仄平仄" 在位置 2,3,4
                    if sim_len >= 5:
                        pz2_gu = sheng_map.get(simulated_line[2], [])
                        pz4 = sheng_map.get(simulated_line[4], [])
                        if len(pz2_gu) == 1 and len(pz4) == 1:
                            if pz2_gu[0] == 1 and pz4[0] == 1:
                                return -1000

        # --- 句尾平仄规则：奇句仄收，偶句平收 (平韵) ---
        if sim_len == target_len:
            last_char = simulated_line[-1]
            pz_last = self._get_pingze(last_char)
            if len(pz_last) == 1:
                expected_tone = "平" if end_tone == 0 else "仄"
                if pz_last[0] != expected_tone:
                    return -1000

        # --- 押韵一致性检查 (行末 + 押韵句) ---
        if sim_len == target_len and is_rhyming:
            last_char = simulated_line[-1]
            expected_tone = "平" if "平" in rhyme_type else "仄"
            rhyme_parts = self._get_rhyme_parts(last_char, expected_tone)
            if locked_rhyme is not None:
                if locked_rhyme not in rhyme_parts:
                    return -1000

        # --- 重复检测 ---
        penalty = 0.0
        # 单字重复 (在原 verifier 中每重复一次 icount -= 3)
        for c in token_chars:
            if c in self.current_line_text:
                penalty += 3.0
            if c in self.all_ci_text:
                penalty += 1.0

        # 2-gram 重复 (硬拒绝，对应原 verifier)
        if len(token_chars) >= 2:
            for i in range(len(token_chars) - 1):
                bigram = token_chars[i:i + 2]
                if bigram in self.current_line_text:
                    return -1000
                if bigram in self.all_ci_text:
                    return -1000
        # 同时检查跨边界 2-gram
        if len(self.current_line_text) >= 1 and len(token_chars) >= 1:
            cross_bigram = self.current_line_text[-1] + token_chars[0]
            if cross_bigram in self.all_ci_text:
                return -1000

        # 检查 token 中是否有字符不在韵书内
        for c in token_chars:
            pz = self._get_pingze(c)
            if not pz:
                return -1000

        return penalty

    def _update_line_tracking(self, latest_text: str):
        """更新当前行文本追踪，并同步更新缓存的平仄基调"""
        for c in latest_text:
            if re.match(r'[，,。.？！；\n]', c):
                self.current_line_text = ""
                self._cached_line_tone = 2
            elif re.match(r'[一-龥A-Za-z]', c):
                self.current_line_text += c
                self.all_ci_text += c
        # 从已生成的行文本确定平仄基调（第2字决定）
        if self._cached_line_tone == 2 and len(self.current_line_text) >= 2:
            pz2 = self._get_pingze(self.current_line_text[1])
            if len(pz2) == 1:
                self._cached_line_tone = 0 if pz2[0] == "平" else 1

    def _handle_punctuation(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """处理标点符号生成需求"""
        pos_info = self.state_machine.get_position_info()
        is_rhyming = pos_info["is_rhyming"]
        line_idx = pos_info["current_line"]

        mask = torch.full_like(scores, -float('inf'))

        if is_rhyming or (line_idx + 1) % 2 == 0:
            # 押韵句或偶数句 → 句号
            target_set = self.period_tokens
        else:
            # 奇数句 → 逗号
            target_set = self.comma_tokens

        for tid in target_set:
            mask[:, tid] = scores[:, tid]

        return mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 1. 解码已生成文本
        generated_ids = input_ids[0][self.input_prompt_len:].tolist()
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        raw_text = raw_text.replace(' ', '').replace('\r', '')

        # 2. 等待 [content] 标记
        if not self.has_started_ci:
            if '[content]' in raw_text:
                self.has_started_ci = True
                ci_text = raw_text.split('[content]', 1)[1]
                self.state_machine.advance_state(ci_text)
                self.last_decoded_text = ci_text
                self._update_line_tracking(ci_text)
            return scores

        # 3. 提取正文并更新状态
        ci_text = raw_text.split('[content]', 1)[1] if '[content]' in raw_text else raw_text
        if len(ci_text) > len(self.last_decoded_text):
            latest_chars = ci_text[len(self.last_decoded_text):]
            self.state_machine.advance_state(latest_chars)
            self.last_decoded_text = ci_text
            self._update_line_tracking(latest_chars)

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
                valid_rhyme_tokens = set()
                for (rt, rp), t_ids in self.vocab_indexer.rhyme_tokens.items():
                    if rt == rhyme_type:
                        valid_rhyme_tokens.update(t_ids)
                base_set = base_set.intersection(valid_rhyme_tokens) if valid_rhyme_tokens else base_set
            elif rhyme_req is not None:
                rhyme_type = pz_pattern[-1]
                valid_rhyme_tokens = self.vocab_indexer.rhyme_tokens.get((rhyme_type, rhyme_req), set())
                base_set = base_set.intersection(valid_rhyme_tokens) if valid_rhyme_tokens else set()

            allowed_tokens.update(base_set)

        # 7. 构建掩码并施加 verifier 规则
        mask = torch.full_like(scores, -float('inf'))
        allowed_list = list(allowed_tokens)

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

        mask[0, allowed_tensor] = token_scores
        return mask
