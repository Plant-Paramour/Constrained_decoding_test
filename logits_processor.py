import torch
import math
import re
from transformers import LogitsProcessor, PreTrainedTokenizer
from vocab_indexer import VocabIndexer
from state_machine import GenerationStateMachine, TangPoemStateMachine

class ConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_indexer: VocabIndexer, state_machine: GenerationStateMachine, tokenizer: PreTrainedTokenizer, input_prompt_len: int):
        self.vocab_indexer = vocab_indexer
        self.state_machine = state_machine
        self.tokenizer = tokenizer
        
        self.input_prompt_len = input_prompt_len
        self.last_decoded_text = ""
        self.has_started_ci = False
        self.eos_token_id = tokenizer.eos_token_id
        
        self.punct_token_ids = self._find_punct_tokens()
        self.newline_token_ids = self._find_newline_tokens()
        self.caesura_token_ids = self._find_caesura_tokens()
        self.terminal_punct_token_ids = self._find_terminal_punct_tokens()
        self.terminal_newline_token_ids = self._find_terminal_newline_tokens()
        self.token_id_to_chars = {}

        # 禁止生成 ； token
        self.semicolon_token_ids = set()
        for tid in self.tokenizer.encode('；', add_special_tokens=False):
            self.semicolon_token_ids.add(tid)

        # 常见双字词集合（用于跨句读粘连检测）
        self.common_bigrams = set()
        for tid, text in self.vocab_indexer.token_to_text.items():
            if len(text) == 2:
                self.common_bigrams.add(text)

    def _find_punct_tokens(self) -> dict:
        punct_tokens = {'odd': set(), 'even': set()}
        valid_puncts_odd = ['，', '？', '！', '，\n']
        valid_puncts_even = ['？', '！']
        
        for token_char, token_id in self.tokenizer.get_vocab().items():
            clean_text = self.tokenizer.decode([token_id]).replace(' ', '')
            if clean_text in valid_puncts_odd:
                punct_tokens['odd'].add(token_id)
            if clean_text in valid_puncts_even:
                punct_tokens['even'].add(token_id)
                
        for p in valid_puncts_odd:
            ids = self.tokenizer.encode(p, add_special_tokens=False)
            if ids:
                punct_tokens['odd'].add(ids[0])
        for p in valid_puncts_even:
            ids = self.tokenizer.encode(p, add_special_tokens=False)
            if ids:
                punct_tokens['even'].add(ids[0])
                
        return punct_tokens

    def _find_newline_tokens(self) -> dict:
        newline_tokens = {'odd': set(), 'even': set()}
        valid_newlines_odd = ['\n', '，\n']
        valid_newlines_even = ['\n']
        
        for token_char, token_id in self.tokenizer.get_vocab().items():
            clean_text = self.tokenizer.decode([token_id]).replace(' ', '')
            if clean_text in valid_newlines_odd:
                newline_tokens['odd'].add(token_id)
            if clean_text in valid_newlines_even:
                newline_tokens['even'].add(token_id)
                
        for p in valid_newlines_odd:
            ids = self.tokenizer.encode(p, add_special_tokens=False)
            if ids:
                newline_tokens['odd'].add(ids[0])
        for p in valid_newlines_even:
            ids = self.tokenizer.encode(p, add_special_tokens=False)
            if ids:
                newline_tokens['even'].add(ids[0])
                
        return newline_tokens

    def _find_caesura_tokens(self) -> set:
        caesura_tokens = set()
        for token_char, token_id in self.tokenizer.get_vocab().items():
            clean_text = self.tokenizer.decode([token_id]).replace(' ', '')
            if clean_text == '、':
                caesura_tokens.add(token_id)
        ids = self.tokenizer.encode('、', add_special_tokens=False)
        if ids:
            caesura_tokens.add(ids[0])
        return caesura_tokens

    def _find_terminal_punct_tokens(self) -> set:
        terminal_puncts = set()
        valid = ['。', '？', '！', '。\n']
        for token_char, token_id in self.tokenizer.get_vocab().items():
            clean_text = self.tokenizer.decode([token_id]).replace(' ', '')
            if clean_text in valid:
                terminal_puncts.add(token_id)
        for p in valid:
            ids = self.tokenizer.encode(p, add_special_tokens=False)
            if ids:
                terminal_puncts.add(ids[0])
        return terminal_puncts
        
    def _find_terminal_newline_tokens(self) -> set:
        terminal_newlines = set()
        valid = ['\n', '。\n']
        for token_char, token_id in self.tokenizer.get_vocab().items():
            clean_text = self.tokenizer.decode([token_id]).replace(' ', '')
            if clean_text in valid:
                terminal_newlines.add(token_id)
        for p in valid:
            ids = self.tokenizer.encode(p, add_special_tokens=False)
            if ids:
                terminal_newlines.add(ids[0])
        return terminal_newlines

    def _get_boundary_positions(self) -> set:
        """从当前行的格律模式中提取句读边界位置（/ 分割点）"""
        line_pattern = self.state_machine.get_current_line_info()
        if not line_pattern:
            return set()
        parts = line_pattern.split('/')
        positions = set()
        accumulated = 0
        for part in parts[:-1]:
            accumulated += len(part)
            positions.add(accumulated)
        return positions

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 全局禁止 ； token
        if self.semicolon_token_ids:
            for tid in self.semicolon_token_ids:
                if tid < scores.shape[1]:
                    scores[:, tid] = -float('inf')

        # 1. 识别出当前生成的进展，更新状态机
        generated_ids = input_ids[0][self.input_prompt_len:].tolist()
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).replace(' ', '').replace('\r', '')

        # 寻找 [content] 标志。一旦找到，立即将之后所有的【中文字符及字母】送入状态机
        if not getattr(self, 'has_started_ci', False):
            if '[content]' in raw_text:
                self.has_started_ci = True
                ci_text = raw_text.split('[content]', 1)[1]
                if len(ci_text) > 0:
                    self.state_machine.advance_state(ci_text)
                self.last_decoded_text = ci_text
            else:
                # 还没遇到 [content]，自由生成，不加约束
                return scores
        else:
            ci_text = raw_text.split('[content]', 1)[1] if '[content]' in raw_text else raw_text
            if len(ci_text) > len(self.last_decoded_text):
                latest_char = ci_text[len(self.last_decoded_text):]
                self.state_machine.advance_state(latest_char)
                self.last_decoded_text = ci_text

        # 2. 如果已经生成完毕，只允许输出 EOS token
        if self.state_machine.is_finished:
            mask = torch.full_like(scores, -float('inf'))
            mask[:, self.eos_token_id] = scores[:, self.eos_token_id]
            return mask

        # NEW: 处理换行符号生成要求
        if getattr(self.state_machine, 'needs_newline', False):
            mask = torch.full_like(scores, -float('inf'))
            if getattr(self.state_machine, 'is_current_line_rhyming', False):
                target_set = self.terminal_newline_token_ids
            else:
                is_odd_line = (self.state_machine.current_line % 2 == 0)
                target_set = self.newline_token_ids['odd'] if is_odd_line else self.newline_token_ids['even']
            allowed_tensor = torch.tensor(list(target_set), dtype=torch.long, device=scores.device)
            if len(allowed_tensor) > 0:
                mask[0, allowed_tensor] = scores[0, allowed_tensor]
            else:
                print(f"[Warning] No newline tokens matched.")
                mask[:, self.eos_token_id] = scores[:, self.eos_token_id]
            return mask

        # NEW: 处理固定位置顿号生成要求
        if getattr(self.state_machine, 'needs_caesura', False):
            mask = torch.full_like(scores, -float('inf'))
            allowed_tensor = torch.tensor(list(self.caesura_token_ids), dtype=torch.long, device=scores.device)
            if len(allowed_tensor) > 0:
                mask[0, allowed_tensor] = scores[0, allowed_tensor]
            else:
                print(f"[Warning] No caesura tokens matched.")
                mask[:, self.eos_token_id] = scores[:, self.eos_token_id]
            return mask

        # NEW: 处理标点符号生成要求
        if getattr(self.state_machine, 'needs_punctuation', False):
            mask = torch.full_like(scores, -float('inf'))
            if getattr(self.state_machine, 'is_current_line_rhyming', False):
                target_set = self.terminal_punct_token_ids
            else:
                is_odd_line = (self.state_machine.current_line % 2 == 0) # 0-indexed, so 0 is line 1 (odd)
                target_set = self.punct_token_ids['odd'] if is_odd_line else self.punct_token_ids['even']
            allowed_tensor = torch.tensor(list(target_set), dtype=torch.long, device=scores.device)
            if len(allowed_tensor) > 0:
                mask[0, allowed_tensor] = scores[0, allowed_tensor]
            else:
                print(f"[Warning] No punctuation tokens matched.")
                mask[:, self.eos_token_id] = scores[:, self.eos_token_id]
            return mask

        # 3. 获取合法模式
        allowed_patterns = self.state_machine.get_allowed_patterns()
        allowed_tokens = set()
        
        has_any_pattern_allowed = False
        for length, pz_pattern, rhyme_req in allowed_patterns:
            # 去索引里查出所有符合 (length, pz_pattern) 的 token ids
            base_set = self.vocab_indexer.pattern_tokens.get((length, pz_pattern), set())
            
            if rhyme_req == "ANY_RHYME":
                # 选择一个有押韵的字
                 # 押韵类型 (平/仄) 基于我们的 json 配置, current char pz pattern last already filtered in state_machine
                 rhyme_type = pz_pattern[-1] # 平/仄
                 valid_rhyme_tokens = set()
                 for (rt, rp), t_ids in self.vocab_indexer.rhyme_tokens.items():
                     if rt == rhyme_type:
                         valid_rhyme_tokens.update(t_ids)
                 base_set = base_set.intersection(valid_rhyme_tokens)
            elif rhyme_req is not None:
                 # 特定韵部
                 rhyme_type = pz_pattern[-1] # 平/仄
                 valid_rhyme_tokens = self.vocab_indexer.rhyme_tokens.get((rhyme_type, rhyme_req), set())
                 base_set = base_set.intersection(valid_rhyme_tokens)
                 
            if base_set:
                has_any_pattern_allowed = True
            allowed_tokens.update(base_set)
            
        # 4. 创建掩码过滤原始 scores
        mask = torch.full_like(scores, -float('inf'))
        allowed_list = list(allowed_tokens)
        allowed_tensor = torch.tensor(allowed_list, dtype=torch.long, device=scores.device)
        
        # 收集已经被生成过的存在于正文中的汉字及其最后出现的绝对位置
        char_to_last_pos = {}
        if getattr(self, 'has_started_ci', False) and '[content]' in raw_text:
            ci_text_for_penalty = raw_text.split('[content]', 1)[1]
        else:
            ci_text_for_penalty = ""

        current_pos = 0
        for c in ci_text_for_penalty:
            if re.match(r'[\u4e00-\u9fa5A-Za-z]', c):
                char_to_last_pos[c] = current_pos
                current_pos += 1
                
        if len(allowed_tensor) > 0:
            token_scores = scores[0, allowed_tensor].clone()
            
            # 指数衰减重复字惩罚逻辑
            if char_to_last_pos:
                # 调整：增加基础保底惩罚，减缓衰减速度
                # 让长距离的重复也付出高昂代价
                base_penalty = 20.0
                decay_rate = 0.05  # 从 0.2 降为 0.05，让惩罚能探到更远的地方
                min_penalty = 8.0  # 增加保底惩罚，任何重复的字至少受到 8.0 的降权

                for idx, t_id in enumerate(allowed_list):
                    if t_id not in self.token_id_to_chars:
                        clean_text = self.tokenizer.decode([t_id]).replace(' ', '')
                        self.token_id_to_chars[t_id] = [c for c in clean_text if re.match(r'[\u4e00-\u9fa5A-Za-z]', c)]
                    
                    token_chars = self.token_id_to_chars[t_id]
                    # 计算当前 token 累加的衰减惩罚
                    token_penalty = 0.0
                    for c in token_chars:
                        if c in char_to_last_pos:
                            distance = current_pos - char_to_last_pos[c]
                            # 防止异常距离，距离最小为 1
                            distance = max(1, distance)
                            
                            # 指数衰减 + 保底惩罚
                            current_penalty = base_penalty * math.exp(-decay_rate * distance)
                            token_penalty += max(min_penalty, current_penalty)
                            
                    if token_penalty > 0:
                        token_scores[idx] -= token_penalty

                # 跨句读 2-gram 粘连检测（移植自 TangPoemLogitsProcessor 的 _verifier_check 逻辑）
                boundary_positions = self._get_boundary_positions()
                line_text = self.state_machine.current_line_text
                if self.state_machine.current_char_idx in boundary_positions and line_text:
                    for idx, t_id in enumerate(allowed_list):
                        if t_id not in self.token_id_to_chars:
                            clean_text = self.tokenizer.decode([t_id]).replace(' ', '')
                            self.token_id_to_chars[t_id] = [c for c in clean_text if re.match(r'[一-龥A-Za-z]', c)]
                        token_chars = self.token_id_to_chars[t_id]
                        if token_chars:
                            bigram = line_text[-1] + token_chars[0]
                            if bigram in self.common_bigrams:
                                token_scores[idx] -= 50.0

            mask[0, allowed_tensor] = token_scores
        else:
            if not has_any_pattern_allowed and len(allowed_patterns) > 0:
                print(f"[Warning] Failed to find valid token matching any allowed pattern and rhyme criteria. Forcing EOS.")
            # 如果没有哪怕任何一个中文字能塞进去，兜底给 eos_token，提前结束或引发异常视情况而定
            mask[:, self.eos_token_id] = scores[:, self.eos_token_id]

        return mask


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

        # 诗成后的状态控制
        self._poem_done = False            # 诗词正文已结束，等待换行或终止
        self._constraints_released = False  # 约束已释放（换行后自由生成解释）

        # 标点 token 缓存
        self._init_punct_tokens()

        # token_id → 解码字符缓存
        self.token_id_to_chars = {}

        # 禁止生成 ； token
        self.semicolon_token_ids = set()
        for tid in self.tokenizer.encode('；', add_special_tokens=False):
            self.semicolon_token_ids.add(tid)

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
        self.question_tokens = set()
        self.exclamation_tokens = set()
        self.newline_tokens = set()
        for c in ['，', ',']:
            for tid in self.tokenizer.encode(c, add_special_tokens=False):
                self.comma_tokens.add(tid)
        for c in ['。', '.']:
            for tid in self.tokenizer.encode(c, add_special_tokens=False):
                self.period_tokens.add(tid)
        for c in ['？', '?']:
            for tid in self.tokenizer.encode(c, add_special_tokens=False):
                self.question_tokens.add(tid)
        for c in ['！', '!']:
            for tid in self.tokenizer.encode(c, add_special_tokens=False):
                self.exclamation_tokens.add(tid)
        for c in ['\n']:
            for tid in self.tokenizer.encode(c, add_special_tokens=False):
                self.newline_tokens.add(tid)

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

        # --- 句读边界硬拒绝：token 字符不得跨越句读断点 ---
        cur_pos = pos_info["current_char_idx"]
        new_pos = cur_pos + len(token_chars)
        for bp in self._boundary_positions:
            if cur_pos < bp < new_pos:
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

        # --- 提前三连同预防：倒数第二字不得与倒数第三、句尾形成三连同 ---
        # 当生成到 target_len-1 位置时，若倒数第三字与句尾强制同调，
        # 则倒数第二字必须为反调，否则末位将无合法候选。
        if sim_len == target_len - 1 and sim_len >= 2 and end_tone != 2:
            third_from_end = simulated_line[target_len - 3] if len(simulated_line) >= target_len - 2 else None
            if third_from_end is not None:
                pz_third = self._get_pingze(third_from_end)
                if len(pz_third) == 1:
                    pz_current = self._get_pingze(simulated_line[-1])
                    if len(pz_current) == 1:
                        # 倒数第三仄 + 句尾必仄 → 倒数第二不能仄
                        if pz_third[0] == "仄" and end_tone == 1 and pz_current[0] == "仄":
                            return -1000
                        # 倒数第三平 + 句尾必平 → 倒数第二不能平
                        if pz_third[0] == "平" and end_tone == 0 and pz_current[0] == "平":
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
            if locked_rhyme_parts and rhyme_parts:
                if not locked_rhyme_parts.intersection(rhyme_parts):
                    return -1000
            # 首句仄收时排除该韵部，后续押韵句不得使用
            excluded_rhyme = pos_info.get("excluded_rhyme_parts")
            if excluded_rhyme and rhyme_parts:
                if excluded_rhyme.intersection(rhyme_parts):
                    return -1000

        # --- 重复检测 ---
        penalty = 0.0
        for c in token_chars:
            if c in line_text:
                penalty += 30.0  # 句内重字严重惩罚
            all_count = self.all_ci_text.count(c)
            if all_count >= 4:
                return -1000  # 全诗同一字出现 4 次以上，硬拒绝
            elif all_count >= 2:
                penalty += all_count * 15.0  # 2-3 次出现，递增惩罚
            elif all_count == 1:
                penalty += 5.0  # 首次重复，温和惩罚

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
                penalty += 50.0  # 跨句读常见词严重破坏节奏

        return penalty

    def _critical_checks(self, token_id: int, pos_info: dict) -> float:
        """兜底时的核心格律底线：仅检查字数、句读、三连同、句尾平仄、二四六分明。

        当所有候选都被 _verifier_check 硬拒绝时调用，放宽次要约束（重复、禁字、双字词等），
        但绝不允许产出明显破律的字。
        """
        target_len = pos_info["target_length"]
        is_rhyming = pos_info["is_rhyming"]
        rhyme_type = pos_info["rhyme_type"]
        line_tone = pos_info.get("base_tone", 2)

        raw_decode = self.tokenizer.decode([token_id])
        if re.search(r'[\s　，。、？！；：\n\r]', raw_decode):
            return -1000

        token_text = self._decode_token(token_id)
        token_chars = self._get_chars_only(token_text)
        if not token_chars:
            return -1000

        line_text = self.state_machine.current_line_text
        cur_pos = pos_info["current_char_idx"]
        new_pos = cur_pos + len(token_chars)
        for bp in self._boundary_positions:
            if cur_pos < bp < new_pos:
                return -1000

        simulated_line = line_text + token_chars
        sim_len = len(simulated_line)

        if sim_len > target_len:
            return -1000

        if line_tone == 2 and sim_len >= 2 and len(line_text) < 2:
            pz2 = self._get_pingze(simulated_line[1])
            if len(pz2) == 1:
                line_tone = 0 if pz2[0] == "平" else 1

        if is_rhyming:
            end_tone = 0 if "平" in rhyme_type else 1
        elif pos_info.get("current_line", -1) == 0:
            end_tone = 2
        else:
            end_tone = 1 if "平" in rhyme_type else 0

        # 二四六分明
        if sim_len >= 2:
            pz = self._get_pingze(simulated_line[1])
            if line_tone != 2 and len(pz) == 1:
                expected = "平" if line_tone == 0 else "仄"
                if pz[0] != expected:
                    return -1000
        if sim_len >= 4:
            pz = self._get_pingze(simulated_line[3])
            if line_tone != 2 and len(pz) == 1:
                expected = "仄" if line_tone == 0 else "平"
                if pz[0] != expected:
                    return -1000
        if sim_len >= 6:
            pz = self._get_pingze(simulated_line[5])
            if line_tone != 2 and len(pz) == 1:
                expected = "平" if line_tone == 0 else "仄"
                if pz[0] != expected:
                    return -1000

        # 提前三连同预防
        if sim_len == target_len - 1 and sim_len >= 2 and end_tone != 2:
            third_from_end = simulated_line[target_len - 3] if len(simulated_line) >= target_len - 2 else None
            if third_from_end is not None:
                pz_third = self._get_pingze(third_from_end)
                if len(pz_third) == 1:
                    pz_current = self._get_pingze(simulated_line[-1])
                    if len(pz_current) == 1:
                        if pz_third[0] == "仄" and end_tone == 1 and pz_current[0] == "仄":
                            return -1000
                        if pz_third[0] == "平" and end_tone == 0 and pz_current[0] == "平":
                            return -1000

        # 三连同
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

        # 句尾平仄
        if sim_len == target_len:
            last_char = simulated_line[-1]
            pz_last = self._get_pingze(last_char)
            if len(pz_last) == 1 and end_tone != 2:
                expected_tone = "平" if end_tone == 0 else "仄"
                if pz_last[0] != expected_tone:
                    return -1000

        # 押韵一致性（核心格律底线，兜底时也必须强制）
        if sim_len == target_len and is_rhyming:
            locked_parts = pos_info.get("locked_rhyme_parts")
            if locked_parts:
                last_char = simulated_line[-1]
                expected_tone = "平" if "平" in rhyme_type else "仄"
                rhyme_parts = set(self._get_rhyme_parts(last_char, expected_tone))
                if rhyme_parts and not locked_parts.intersection(rhyme_parts):
                    return -1000

        return 0.0

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
        # 偶数句（押韵句）允许 。？！ 由模型根据语境自选；奇数句用逗号
        if (line_idx + 1) % 2 == 0:
            target_set = self.period_tokens | self.question_tokens | self.exclamation_tokens
        else:
            target_set = self.comma_tokens
        # 安全兜底：若标点 token 集为空（极端情况），放行全部原始分数，避免全 -inf
        if not target_set:
            return scores
        mask = torch.full_like(scores, -float('inf'))
        for tid in target_set:
            mask[:, tid] = scores[:, tid]
        return mask

    # ── HuggingFace LogitsProcessor 接口 ──────────────────────

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 全局禁止 ； token
        if self.semicolon_token_ids:
            for tid in self.semicolon_token_ids:
                if tid < scores.shape[1]:
                    scores[:, tid] = -float('inf')

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

        # 4. 约束已释放（诗成换行后进入自由解释模式，大模型可自由发挥）
        if self._constraints_released:
            return scores

        # 5. 诗成等待换行 → 仅允许 \n 或 EOS（最多两 token 的过渡窗口）
        if self._poem_done:
            if generated_ids:
                last_text = self.tokenizer.decode([generated_ids[-1]])
                if '\n' in last_text:
                    self._constraints_released = True
                    return scores
            mask = torch.full_like(scores, -float('inf'))
            for tid in self.newline_tokens:
                mask[:, tid] = scores[:, tid]
            mask[:, self.eos_token_id] = scores[:, self.eos_token_id]
            return mask

        # 6. 状态机报告诗成 → 进入诗成等待态，禁止 EOS 提前终止
        if self.state_machine.is_finished:
            self._poem_done = True
            mask = torch.full_like(scores, -float('inf'))
            for tid in self.newline_tokens:
                mask[:, tid] = scores[:, tid]
            mask[:, self.eos_token_id] = scores[:, self.eos_token_id]
            return mask

        # 7. 需要标点 → 输出标点
        if self.state_machine.needs_punctuation:
            return self._handle_punctuation(scores)

        # 8. 获取合法模式并收集候选 token
        allowed_patterns = self.state_machine.get_allowed_patterns()
        allowed_tokens = set()
        for length, pz_pattern, rhyme_req in allowed_patterns:
            base_set = self.vocab_indexer.pattern_tokens.get((length, pz_pattern), set())
            if rhyme_req == "ANY_RHYME":
                rhyme_type = pz_pattern[-1]
                valid_rhyme = set()
                excluded_parts = self.state_machine.excluded_rhyme_parts or set()
                for (rt, rp), t_ids in self.vocab_indexer.rhyme_tokens.items():
                    if rt == rhyme_type and rp not in excluded_parts:
                        valid_rhyme.update(t_ids)
                if valid_rhyme:
                    base_set = base_set.intersection(valid_rhyme)
            elif rhyme_req is not None:
                rhyme_type = pz_pattern[-1]
                valid_rhyme = self.vocab_indexer.rhyme_tokens.get((rhyme_type, rhyme_req), set())
                if valid_rhyme:
                    base_set = base_set.intersection(valid_rhyme)
            allowed_tokens.update(base_set)

        # 9. 构建掩码并施加 verifier 规则
        mask = torch.full_like(scores, -float('inf'))
        vocab_size = scores.shape[1]

        # 过滤越界 token（vocab_indexer 的 token ID 可能超出模型实际 vocab 范围）
        allowed_list = [tid for tid in allowed_tokens if 0 <= tid < vocab_size]
        if not allowed_list:
            # 韵部约束太严格 → 放宽：只用平仄模式匹配，不限定韵部
            allowed_tokens_relaxed = set()
            for length, pz_pattern, rhyme_req in allowed_patterns:
                base_set = self.vocab_indexer.pattern_tokens.get((length, pz_pattern), set())
                allowed_tokens_relaxed.update(base_set)
            allowed_list = [tid for tid in allowed_tokens_relaxed if 0 <= tid < vocab_size]
            if not allowed_list:
                # 最终兜底：允许除 EOS 外全部 token，绝不放行 EOS
                mask[:, :] = scores[:, :]
                mask[:, self.eos_token_id] = -float('inf')
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

        # 所有候选均被硬拒绝 → 放宽次要规则，但保留核心格律底线
        if (token_scores == -float('inf')).all():
            for idx, t_id in enumerate(allowed_list):
                critical_penalty = self._critical_checks(t_id, pos_info)
                if critical_penalty <= -100:
                    token_scores[idx] = -float('inf')
                else:
                    token_scores[idx] = scores[0, t_id] - critical_penalty

        # 核心底线仍全部拒绝 → 最终兜底：放行全部原始分数，绝不让 mask 全 -inf
        if (token_scores == -float('inf')).all():
            mask[:, :] = scores[:, :]
            return mask

        mask[0, allowed_tensor] = token_scores
        return mask

