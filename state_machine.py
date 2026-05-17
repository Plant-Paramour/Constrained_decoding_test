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
        self._current_line_text = ""

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

    @property
    def current_line_text(self) -> str:
        """当前行已生成的文本（供 LogitsProcessor 做跨句读粘连检测）"""
        return self._current_line_text

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
                self._current_line_text = ""

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
                self._current_line_text = ""

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
        self._current_line_text += text

        if self.current_char_idx >= target_len:
            self.current_char_idx = 0
            if self.current_line == self.stanzas[self.current_stanza]["num_lines"] - 1:
                self.needs_newline = True
            else:
                self.needs_punctuation = True


class TangPoemStateMachine:
    """唐诗专用生成状态机 —— 纯算法驱动，不依赖格律 JSON

    基于 poem_verifier 风格的二四六分明规则，由参数（五言/七言、绝句/律诗）
    动态决定每位置允许的平仄模式。平仄基调从已生成文本的第2字实时确定。
    """

    def __init__(self, line_length: int, num_lines: int, rhyme_type: str,
                 data_manager: DataManager):
        if line_length not in (5, 7):
            raise ValueError(f"唐诗仅支持五言(5)或七言(7)，收到: {line_length}")
        if num_lines not in (4, 8):
            raise ValueError(f"唐诗仅支持绝句(4)或律诗(8)，收到: {num_lines}")

        self.line_length = line_length      # 5 或 7
        self.num_lines = num_lines          # 4 或 8
        self.rhyme_type = rhyme_type.strip()  # "平韵" 或 "仄韵"
        self.data_manager = data_manager

        self.current_line = 0       # 0-indexed
        self.current_char_idx = 0   # 当前句已生成字数
        self.locked_rhyme_parts: Optional[set] = None
        self.excluded_rhyme_parts: Optional[set] = None  # 首句仄收时排除的韵部
        self.needs_punctuation = False
        self.is_finished = False

        # 当前行已生成的文本（用于确定平仄基调）
        self._current_line_text = ""
        self._cached_base_tone = 2  # 0=平起, 1=仄起, 2=未定

        # 全诗平仄基调（首句第2字确定后不再改变，带第4/6字回退）
        self._global_base_tone = 2  # 0=平起, 1=仄起, 2=未定
        # 首句是否入韵（末字平声则入韵）
        self._line0_rhymes = False

    # ── 公开查询 ──────────────────────────────────────────────

    def get_target_length(self) -> int:
        return self.line_length

    def _is_rhyming_line(self) -> bool:
        """偶数句押韵（绝句第2、4句；律诗第2、4、6、8句），首句末字平声则入韵"""
        if self.current_line == 0:
            return self._line0_rhymes
        return (self.current_line + 1) % 2 == 0

    def _get_expected_end_tone(self) -> str:
        """押韵句末字与韵式同调，非押韵句相反"""
        if self._is_rhyming_line():
            return "平" if "平" in self.rhyme_type else "仄"
        else:
            return "仄" if "平" in self.rhyme_type else "平"

    # ── 核心：动态平仄模式生成 ────────────────────────────────

    def _get_allowed_pingze_at(self, pos_idx: int) -> List[str]:
        """返回 pos_idx（0-indexed）位置允许的平仄列表，基于二四六分明规则"""
        base = self._cached_base_tone  # 0=平, 1=仄, 2=未定

        if base == 2:
            # 基调尚未确定，所有位置自由
            return ["平", "仄"]

        # 二四六分明
        if pos_idx == 1 or (pos_idx == 5 and self.line_length >= 7):
            # 第2字 / 第6字：必须与基调相同
            return ["平"] if base == 0 else ["仄"]
        elif pos_idx == 3:
            # 第4字：必须与基调相反
            return ["仄"] if base == 0 else ["平"]
        else:
            # 一三五不论
            return ["平", "仄"]

    def get_allowed_patterns(self, max_length: int = 4) -> List[Tuple[int, str, Optional[str]]]:
        """返回当前步允许的 (字数, 平仄模式, 韵部要求) 列表"""
        if self.is_finished or self.needs_punctuation:
            return []

        target_len = self.line_length
        remains = target_len - self.current_char_idx
        allowed = []

        # ── 句读：构建虚拟断句模式并计算距下一断点的剩余字数 ──
        # 五言 "2/3"，七言 "2/2/3"
        if self.line_length == 5:
            virtual_pattern = "??/???"
        else:  # 7
            virtual_pattern = "??/??/???"

        parts = virtual_pattern.split('/')
        accumulated = 0
        remains_before_break = remains
        for part in parts:
            part_len = len(part)
            if self.current_char_idx < accumulated + part_len:
                remains_before_break = accumulated + part_len - self.current_char_idx
                break
            accumulated += part_len

        is_rhyme_line = self._is_rhyming_line()
        expected_end_tone = self._get_expected_end_tone()

        for L in range(1, min(max_length, remains_before_break) + 1):
            # 生成接下来 L 个字符的所有合法平仄组合
            pz_combos = [""]
            for offset in range(L):
                pos_allowed = self._get_allowed_pingze_at(self.current_char_idx + offset)
                new_combos = []
                for pz in pos_allowed:
                    for combo in pz_combos:
                        new_combos.append(combo + pz)
                pz_combos = new_combos

            if L == remains:
                if is_rhyme_line:
                    # 押韵句末：末字平仄必须匹配韵式
                    for pz in pz_combos:
                        if not pz:
                            continue
                        if "平" in self.rhyme_type and pz[-1] != "平":
                            continue
                        if "仄" in self.rhyme_type and pz[-1] != "仄":
                            continue
                        if self.locked_rhyme_parts:
                            for rp in self.locked_rhyme_parts:
                                allowed.append((L, pz, rp))
                        else:
                            allowed.append((L, pz, "ANY_RHYME"))
                elif self.current_line == 0:
                    # 首句末字可平可仄（由实际生成结果决定是否入韵）
                    for pz in pz_combos:
                        if pz:
                            allowed.append((L, pz, None))
                else:
                    # 非押韵句末：句尾平仄必须符合奇偶规则
                    for pz in pz_combos:
                        if pz and pz[-1] == expected_end_tone:
                            allowed.append((L, pz, None))
            else:
                for pz in pz_combos:
                    if pz:
                        allowed.append((L, pz, None))

        return allowed

    # ── 状态推进 ──────────────────────────────────────────────

    def advance_state(self, text: str):
        if self.is_finished or not text:
            return

        import re

        # 处理标点需求
        if self.needs_punctuation:
            has_punct = any(re.match(r'[，。？！；\n]', c) for c in text)
            if has_punct:
                self.needs_punctuation = False
                self.current_line += 1
                self._current_line_text = ""
                # 从全局基调推导本行基调（ids 0,3,4,7 同调, ids 1,2,5,6 反弹）
                if self._global_base_tone != 2:
                    if self.current_line in (1, 2, 5, 6):
                        self._cached_base_tone = 1 - self._global_base_tone
                    else:
                        self._cached_base_tone = self._global_base_tone
                else:
                    self._cached_base_tone = 2
                if self.current_line >= self.num_lines:
                    self.is_finished = True
                self.current_char_idx = 0
                return

        # 提取有效汉字
        valid_chars = "".join(
            c for c in text if re.match(r'[一-龥A-Za-z]', c)
        )
        if not valid_chars:
            return

        length = len(valid_chars)

        # 确定本句平仄基调：优先第2字，多音字则回退到第4字，再回退到第6字
        if self._cached_base_tone == 2:
            tone = 2
            # 尝试第2字 (idx 1)
            pos2_in_new = 1 - self.current_char_idx
            if 0 <= pos2_in_new < len(valid_chars):
                char2 = valid_chars[pos2_in_new]
                pz = self.data_manager.get_pingze(char2)
                if len(pz) == 1:
                    tone = 0 if pz[0] == "平" else 1
            # 回退：第4字 (idx 3)，与基调相反
            if tone == 2:
                pos4_in_new = 3 - self.current_char_idx
                if 0 <= pos4_in_new < len(valid_chars):
                    char4 = valid_chars[pos4_in_new]
                    pz = self.data_manager.get_pingze(char4)
                    if len(pz) == 1:
                        tone = 1 if pz[0] == "平" else 0
            # 回退：第6字 (idx 5)，与基调相同（仅七言）
            if tone == 2 and self.line_length >= 7:
                pos6_in_new = 5 - self.current_char_idx
                if 0 <= pos6_in_new < len(valid_chars):
                    char6 = valid_chars[pos6_in_new]
                    pz = self.data_manager.get_pingze(char6)
                    if len(pz) == 1:
                        tone = 0 if pz[0] == "平" else 1
            if tone != 2:
                self._cached_base_tone = tone
                if self.current_line == 0 and self._global_base_tone == 2:
                    self._global_base_tone = tone

        # 押韵句末 — 锁定/更新韵部（交集策略，与原 GLM 逻辑一致）
        if self.current_char_idx + length == self.line_length:
            if self._is_rhyming_line():
                last_char = valid_chars[-1]
                expected_tone = "平" if "平" in self.rhyme_type else "仄"
                rhyme_parts = self.data_manager.get_rhyme_part_by_tone(last_char, expected_tone)
                if rhyme_parts:
                    if self.locked_rhyme_parts is None:
                        self.locked_rhyme_parts = set(rhyme_parts)
                    else:
                        new_parts = self.locked_rhyme_parts.intersection(set(rhyme_parts))
                        if new_parts:
                            self.locked_rhyme_parts = new_parts
                        # 交集为空则保持旧锁，防止锁变为空集合导致押韵约束失效
            elif self.current_line == 0 and not self._line0_rhymes:
                # 首句末字平声则入韵（模仿原 verify_rhy: ids==0 且末字平声 → needyy=1）
                last_char = valid_chars[-1]
                pz_last = self.data_manager.get_pingze(last_char)
                if len(pz_last) == 1 and "平" in self.rhyme_type and pz_last[0] == "平":
                    self._line0_rhymes = True
                    expected_tone = "平"
                    rhyme_parts = self.data_manager.get_rhyme_part_by_tone(last_char, expected_tone)
                    if rhyme_parts:
                        self.locked_rhyme_parts = set(rhyme_parts)
                elif len(pz_last) == 1 and pz_last[0] == "仄":
                    # 首句仄收 → 排除该字的韵部，第二句绝不能与之押韵
                    rhyme_parts = self.data_manager.get_rhyme_part(last_char)
                    if rhyme_parts:
                        self.excluded_rhyme_parts = set(rhyme_parts)

        self.current_char_idx += length
        self._current_line_text += valid_chars

        if self.current_char_idx >= self.line_length:
            self.current_char_idx = 0
            self.needs_punctuation = True

    def get_position_info(self) -> dict:
        return {
            "current_line": self.current_line,
            "current_char_idx": self.current_char_idx,
            "target_length": self.line_length,
            "is_rhyming": self._is_rhyming_line(),
            "locked_rhyme_parts": self.locked_rhyme_parts,
            "excluded_rhyme_parts": self.excluded_rhyme_parts,
            "rhyme_type": self.rhyme_type,
            "base_tone": self._cached_base_tone,
            "global_base_tone": self._global_base_tone,
            "line0_rhymes": self._line0_rhymes,
            "needs_punctuation": self.needs_punctuation,
            "is_finished": self.is_finished,
        }

    @property
    def current_line_text(self) -> str:
        """暴露当前行文本给 LogitsProcessor 做细粒度校验"""
        return self._current_line_text
