from typing import List, Tuple, Optional, Set
from data_manager import DataManager

class GenerationStateMachine:
    def __init__(self, cipai_name: str, data_manager: DataManager):
        self.cipai = data_manager.get_cipai(cipai_name)
        self.data_manager = data_manager

        self.stanzas = [self.cipai[f"stanza{i+1}"] for i in range(self.cipai["number_of_stanzas"])]
        
        self.current_stanza = 0
        self.current_line = 0
        self.current_char_idx = 0  # 当前句已经生成了多少个字
        
        self.needs_punctuation = False
        self.needs_newline = False
        
        self.locked_rhyme_parts = {}
        self.rhyme_type = self.cipai.get("rhyme_type", "平韵").strip()
        
        self.is_finished = False

    def get_total_length(self) -> int:
        """获取当前词牌的正文总字数"""
        total = 0
        for stanza in self.stanzas:
            for line_pattern in stanza.get("lines", []):
                total += len(line_pattern.replace("/", ""))
        return total

    @property
    def needs_caesura(self) -> bool:
        """检查当前字符是否为固定的顿号（如 桂枝香 里的“仄平平、中中/平仄”）"""
        if self.is_finished:
            return False
        # 如果即将换行或者正常的句尾标点，暂时不认定为 needs_caesura
        if getattr(self, 'needs_punctuation', False) or getattr(self, 'needs_newline', False):
            return False
        pure_pattern = self.get_current_line_info().replace("/", "")
        if self.current_char_idx < len(pure_pattern):
            return pure_pattern[self.current_char_idx] == "、"
        return False

    def get_current_line_info(self):
        if self.is_finished:
            return None
        return self.stanzas[self.current_stanza]["lines"][self.current_line]

    def _get_target_length(self):
        # 抛弃依赖 json 中容易出错的 chars_per_line，直接从当前的格律字符串实时计算长度
        line_pattern = self.get_current_line_info()
        return len(line_pattern.replace("/", ""))

    def _get_rhyme_group(self) -> Optional[int]:
        # 检查当前句是否属于某个 rhyme_X_positions，并返回其分组的数字 X。若不是押韵句则返回 None
        stanza = self.stanzas[self.current_stanza]
        line_idx = self.current_line + 1
        for key, value in stanza.items():
            if key.startswith("rhyme") and key.endswith("positions"):
                if line_idx in value:
                    parts = key.split("_")
                    if len(parts) >= 3 and parts[1].isdigit():
                        return int(parts[1])
                    elif key == "rhyme_positions":
                        return 1
        return None

    @property
    def is_current_line_rhyming(self) -> bool:
        return self._get_rhyme_group() is not None

    def get_allowed_patterns(self, max_length: int = 4) -> List[Tuple[int, str, Optional[str]]]:
        if self.is_finished:
            return []
            
        if getattr(self, 'needs_punctuation', False) or getattr(self, 'needs_newline', False):
            return []
            
        if self.needs_caesura:
            return []
            
        line_pattern_raw = self.get_current_line_info()
        pure_pattern = line_pattern_raw.replace("/", "")
        target_len = self._get_target_length()
        
        # 还可以生成多少字
        remains = target_len - self.current_char_idx
        
        # 计算距离下一个断句点(或句末)的剩余字数
        char_count = 0
        remains_before_break = remains
        
        # 修复断句字数逻辑：
        # 我们要找 pure_pattern 下对应的段落
        # 直接通过对 line_pattern_raw 按照 / 分割，找到当前 idx 所在的词块
        parts = line_pattern_raw.split('/')
        accumulated = 0
        for part in parts:
            part_len = len(part)
            if self.current_char_idx < accumulated + part_len:
                remains_before_break = accumulated + part_len - self.current_char_idx
                break
            accumulated += part_len
                
        allowed = []
        
        rhyme_group = self._get_rhyme_group()
        is_rhyme_line = rhyme_group is not None
        
        # 尝试不同长度的 Token, 长度不能超过距离下一个断句点的剩余字数
        for L in range(1, min(max_length, remains_before_break) + 1):
            target_slice = pure_pattern[self.current_char_idx : self.current_char_idx + L]
            
            # 如果截取的切片中包含了 "、"，是不合法的（因为 "、" 必须单独作为标点输出）
            if "、" in target_slice:
                continue

            # 展开"中"字
            expanded_pzs = self._expand_zhong(target_slice)
            
            # 检查是否触及句末押韵位
            if L == remains and is_rhyme_line:
                # 押韵的情况
                for pz in expanded_pzs:
                    if not pz:
                        continue
                    
                    # 只有主韵 (rhyme_group == 1) 才强制校验全局 rhyme_type。其余韵部遵循其句子原本的平仄要求即可。
                    if rhyme_group == 1:
                        if (self.rhyme_type == "平韵" and pz[-1] != "平") or (self.rhyme_type == "仄韵" and pz[-1] != "仄"):
                            continue
                    
                    if rhyme_group in self.locked_rhyme_parts:
                        allowed.append((L, pz, self.locked_rhyme_parts[rhyme_group]))
                    else:
                        # 还没定韵，可以任取 (但在 logits processor 里我们会用所有合规的韵部子集)
                        allowed.append((L, pz, "ANY_RHYME"))
            else:
                for pz in expanded_pzs:
                    if pz:
                        allowed.append((L, pz, None))
                    
        return allowed

    def _expand_zhong(self, pattern: str) -> List[str]:
        results = [""]
        for char in pattern:
            if char == "中":
                new_results = []
                for r in results:
                    new_results.append(r + "平")
                    new_results.append(r + "仄")
                results = new_results
            else:
                results = [r + char for r in results]
        return results

    def advance_state(self, text: str):
        if self.is_finished or not text:
            return
            
        import re
        
        # 将输入分离：标点符号（用于处理 needs_punctuation 或 needs_newline）和纯汉字（用于处理字数）
        # 如果需要换行
        if getattr(self, 'needs_newline', False):
            has_newline = False
            for char in text:
                if char == '\n':
                    has_newline = True
                    break
            
            if has_newline:
                self.needs_newline = False
                self.current_line += 1
                
                if self.current_line >= self.stanzas[self.current_stanza]["num_lines"]:
                    self.current_line = 0
                    self.current_stanza += 1
                    
                    if self.current_stanza >= len(self.stanzas):
                        self.is_finished = True

        # 如果需要标点
        elif getattr(self, 'needs_punctuation', False):
            # 寻找标点
            has_punct = False
            for char in text:
                if re.match(r'[，。、？！；\n]', char):
                    has_punct = True
                    break
            
            if has_punct:
                self.needs_punctuation = False
                self.current_line += 1
                
                if self.current_line >= self.stanzas[self.current_stanza]["num_lines"]:
                    self.current_line = 0
                    self.current_stanza += 1
                    
                    if self.current_stanza >= len(self.stanzas):
                        self.is_finished = True
            
            # 标点处理完后，如果这段 text 里还有汉字（比如模型同时输出了标点和汉字），需要继续处理
            # 截取标点之后的文本继续处理（简单起见，提取所有汉字）
            
        # 如果需要强制顿号输出
        elif self.needs_caesura:
            has_caesura = False
            for char in text:
                if char == '、':
                    has_caesura = True
                    break
            if has_caesura:
                # 只有 current_char_idx 步进，不换句
                self.current_char_idx += 1
                # 可能后面连着汉字，下面统一处理 valid_chars

        # 提取纯汉字（以及放宽的英文字母）部分计算字数和押韵
        valid_chars = ""
        for char in text:
            if re.match(r'[\u4e00-\u9fa5A-Za-z]', char):
                valid_chars += char
                
        if not valid_chars:
            return
            
        text = valid_chars
        
        length = len(text)
        target_len = self._get_target_length()
        
        # 如果是押韵位的第一个定韵字
        rhyme_group = self._get_rhyme_group()
        if self.current_char_idx + length == target_len and rhyme_group is not None:
            if rhyme_group not in self.locked_rhyme_parts:
                last_char = text[-1]
                
                # 确定定韵期望的平仄
                if rhyme_group == 1:
                    expected_tone = "平" if "平" in self.rhyme_type else "仄"
                else:
                    # 对于附加次韵部，从当前这句的格律里获取该字的期望平仄，而不是全局
                    pure_pattern = self.get_current_line_info().replace("/", "")
                    expected_tone = pure_pattern[-1]
                    if expected_tone not in ["平", "仄"]:
                        expected_tone = "仄" if "平" in self.rhyme_type else "平"
                        
                rhyme_parts = self.data_manager.get_rhyme_part_by_tone(last_char, expected_tone)
                if rhyme_parts:
                    self.locked_rhyme_parts[rhyme_group] = rhyme_parts[0] # 贪心取第一个
        
        self.current_char_idx += length
        
        if self.current_char_idx >= target_len:
            self.current_char_idx = 0
            if self.current_line == self.stanzas[self.current_stanza]["num_lines"] - 1:
                self.needs_newline = True
            else:
                self.needs_punctuation = True
