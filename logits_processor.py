import torch
import math
import re
from transformers import LogitsProcessor, PreTrainedTokenizer
from vocab_indexer import VocabIndexer
from state_machine import GenerationStateMachine

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
        self.token_id_to_chars = {}
        
    def _find_punct_tokens(self) -> dict:
        punct_tokens = {'odd': set(), 'even': set()}
        valid_puncts_odd = ['，', '？', '！', '；', '，\n']
        valid_puncts_even = ['。', '？', '！', '；', '。\n']
        
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
        valid_newlines_even = ['\n', '。\n']
        
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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
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
                 # 押韵类型 (平/仄) 基于我们的 json 配置,当前字符 平仄 模式最后已在 state_machine 中过滤
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
                
        # 检查是否为唐诗任务以应用更严格的重复惩罚
        is_tangpoem = hasattr(self, 'is_tangpoem_flag') and self.is_tangpoem_flag

        if len(allowed_tensor) > 0:
            token_scores = scores[0, allowed_tensor].clone()
            
            # 指数衰减重复字惩罚
            if char_to_last_pos:
                base_penalty = 20.0
                decay_rate = 0.05  # 从 0.2 降为 0.05，让惩罚能探到更远的地方
                min_penalty = 8.0  # 增加保底惩罚，任何重复的字至少受到 8.0 的降权
                
                # 如果是唐诗模式，采用绝对的死板惩罚，因为唐诗通常极少复字
                is_strict_tang_penalty = is_tangpoem # 可以自由开关此项
                tang_penalty_val = float('inf') 

                for idx, t_id in enumerate(allowed_list):
                    if t_id not in self.token_id_to_chars:
                        clean_text = self.tokenizer.decode([t_id]).replace(' ', '')
                        self.token_id_to_chars[t_id] = [c for c in clean_text if re.match(r'[\u4e00-\u9fa5A-Za-z]', c)]
                    
                    token_chars = self.token_id_to_chars[t_id]
                    # 计算当前 token 累加的衰减惩罚
                    token_penalty = 0.0
                    has_repeated_char = False
                    for c in token_chars:
                        if c in char_to_last_pos:
                            has_repeated_char = True
                            distance = current_pos - char_to_last_pos[c]
                            # 防止异常距离，距离最小为 1
                            distance = max(1, distance)
                            
                            # 指数衰减 + 保底惩罚
                            current_penalty = base_penalty * math.exp(-decay_rate * distance)
                            token_penalty += max(min_penalty, current_penalty)
                            
                    if is_strict_tang_penalty and has_repeated_char:
                         token_penalty = tang_penalty_val
                    
                    if token_penalty > 0:
                        token_scores[idx] -= token_penalty
                        
            mask[0, allowed_tensor] = token_scores
        else:
            if not has_any_pattern_allowed and len(allowed_patterns) > 0:
                print(f"[Warning] Failed to find valid token matching any allowed pattern and rhyme criteria. Forcing EOS.")
            # 如果没有哪怕任何一个中文字能塞进去，兜底给 eos_token，提前结束或引发异常视情况而定
            mask[:, self.eos_token_id] = scores[:, self.eos_token_id]

        return mask

