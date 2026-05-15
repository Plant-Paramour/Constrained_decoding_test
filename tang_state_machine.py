from typing import List, Tuple, Optional
from data_manager import DataManager


class TangPoemStateMachine:
    """唐诗专用生成状态机 —— 追踪当前行/字位置，提供允许的平仄模式"""

    def __init__(self, poem_form_name: str, data_manager: DataManager):
        self.poem_form = data_manager.get_cipai(poem_form_name)
        self.data_manager = data_manager

        stanza = self.poem_form["stanza1"]
        self.num_lines = stanza["num_lines"]
        self.line_patterns = stanza["lines"]
        self.rhyme_positions = stanza.get("rhyme_positions", [])
        self.rhyme_type = self.poem_form.get("rhyme_type", "平韵").strip()

        self.current_line = 0
        self.current_char_idx = 0
        self.locked_rhyme_parts: Optional[set] = None  # 允许的韵部集合（多韵部交集）
        self.needs_punctuation = False
        self.is_finished = False

        # 预计算每句的目标字数
        self._target_lengths = [
            len(p.replace("/", "")) for p in self.line_patterns
        ]

    def get_target_length(self) -> int:
        return self._target_lengths[self.current_line]

    def get_current_line_pattern(self) -> str:
        return self.line_patterns[self.current_line]

    def _is_rhyming_line(self) -> bool:
        return (self.current_line + 1) in self.rhyme_positions

    def _get_expected_line_end_tone(self) -> str:
        """获取当前句末期望的平仄：奇句仄收，偶句平收（平韵）；反之（仄韵）"""
        if "平" in self.rhyme_type:
            return "平" if (self.current_line + 1) % 2 == 0 else "仄"
        else:
            return "仄" if (self.current_line + 1) % 2 == 0 else "平"

    def get_allowed_patterns(self, max_length: int = 4) -> List[Tuple[int, str, Optional[str]]]:
        if self.is_finished:
            return []
        if self.needs_punctuation:
            return []

        line_pattern_raw = self.get_current_line_pattern()
        pure_pattern = line_pattern_raw.replace("/", "")
        target_len = self.get_target_length()
        remains = target_len - self.current_char_idx

        # 计算距离下一个断句点的剩余字数
        parts = line_pattern_raw.split("/")
        accumulated = 0
        remains_before_break = remains
        for part in parts:
            part_len = len(part)
            if self.current_char_idx < accumulated + part_len:
                remains_before_break = accumulated + part_len - self.current_char_idx
                break
            accumulated += part_len

        allowed = []
        is_rhyme_line = self._is_rhyming_line()

        for L in range(1, min(max_length, remains_before_break) + 1):
            target_slice = pure_pattern[self.current_char_idx:self.current_char_idx + L]
            expanded_pzs = self._expand_zhong(target_slice)

            if L == remains and is_rhyme_line:
                for pz in expanded_pzs:
                    if not pz:
                        continue
                    # 平韵诗押平声韵，仄韵诗押仄声韵
                    if "平" in self.rhyme_type and pz[-1] != "平":
                        continue
                    if "仄" in self.rhyme_type and pz[-1] != "仄":
                        continue
                    if self.locked_rhyme_parts:
                        # 为每个允许的韵部生成一条模式
                        for rp in self.locked_rhyme_parts:
                            allowed.append((L, pz, rp))
                    else:
                        allowed.append((L, pz, "ANY_RHYME"))
            elif L == remains:
                # 非押韵句末 — 确保句尾平仄符合奇偶规则
                expected_tone = self._get_expected_line_end_tone()
                for pz in expanded_pzs:
                    if pz and pz[-1] == expected_tone:
                        allowed.append((L, pz, None))
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

        # 处理标点需求
        if self.needs_punctuation:
            has_punct = any(re.match(r'[，。？！；\n]', c) for c in text)
            if has_punct:
                self.needs_punctuation = False
                self.current_line += 1
                if self.current_line >= self.num_lines:
                    self.is_finished = True
                self.current_char_idx = 0

        # 提取有效汉字推进字数
        valid_chars = "".join(c for c in text if re.match(r'[一-龥A-Za-z]', c))
        if not valid_chars:
            return

        target_len = self.get_target_length()
        length = len(valid_chars)

        # 押韵句末 — 锁定韵部（采用交集策略，与原 GLM 逻辑一致）
        if self.current_char_idx + length == target_len and self._is_rhyming_line():
            last_char = valid_chars[-1]
            expected_tone = "平" if "平" in self.rhyme_type else "仄"
            rhyme_parts = self.data_manager.get_rhyme_part_by_tone(last_char, expected_tone)
            if rhyme_parts:
                if self.locked_rhyme_parts is None:
                    self.locked_rhyme_parts = set(rhyme_parts)
                else:
                    self.locked_rhyme_parts = self.locked_rhyme_parts.intersection(set(rhyme_parts))

        self.current_char_idx += length

        if self.current_char_idx >= target_len:
            self.current_char_idx = 0
            self.needs_punctuation = True

    def get_position_info(self) -> dict:
        """返回当前位置的详细信息，供 LogitsProcessor 做细粒度校验"""
        return {
            "current_line": self.current_line,
            "current_char_idx": self.current_char_idx,
            "target_length": self.get_target_length(),
            "is_rhyming": self._is_rhyming_line(),
            "locked_rhyme_parts": self.locked_rhyme_parts,
            "rhyme_type": self.rhyme_type,
            "line_pattern": self.get_current_line_pattern(),
            "needs_punctuation": self.needs_punctuation,
            "is_finished": self.is_finished,
        }
