from typing import List, Tuple, Optional
from data_manager import DataManager


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
        self.needs_punctuation = False
        self.is_finished = False

        # 当前行已生成的文本（用于确定平仄基调）
        self._current_line_text = ""
        self._cached_base_tone = 2  # 0=平起, 1=仄起, 2=未定

    # ── 公开查询 ──────────────────────────────────────────────

    def get_target_length(self) -> int:
        return self.line_length

    def _is_rhyming_line(self) -> bool:
        """唐诗惯例：偶数句押韵"""
        return (self.current_line + 1) % 2 == 0

    def _get_expected_end_tone(self) -> str:
        """奇句仄收，偶句平收（平韵）；反之（仄韵）"""
        if "平" in self.rhyme_type:
            return "平" if (self.current_line + 1) % 2 == 0 else "仄"
        else:
            return "仄" if (self.current_line + 1) % 2 == 0 else "平"

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

        is_rhyme_line = self._is_rhyming_line()
        expected_end_tone = self._get_expected_end_tone()

        for L in range(1, min(max_length, remains) + 1):
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

        # 确定本句平仄基调：第2字 (idx 1) 生成后即可确定
        if self._cached_base_tone == 2:
            # 判断第2字是否已在本步新文本中
            pos2_in_new = 1 - self.current_char_idx
            if 0 <= pos2_in_new < len(valid_chars):
                char2 = valid_chars[pos2_in_new]
                pz = self.data_manager.get_pingze(char2)
                if len(pz) == 1:
                    self._cached_base_tone = 0 if pz[0] == "平" else 1

        # 押韵句末 — 锁定/更新韵部（交集策略，与原 GLM 逻辑一致）
        if self.current_char_idx + length == self.line_length and self._is_rhyming_line():
            last_char = valid_chars[-1]
            expected_tone = "平" if "平" in self.rhyme_type else "仄"
            rhyme_parts = self.data_manager.get_rhyme_part_by_tone(last_char, expected_tone)
            if rhyme_parts:
                if self.locked_rhyme_parts is None:
                    self.locked_rhyme_parts = set(rhyme_parts)
                else:
                    self.locked_rhyme_parts = self.locked_rhyme_parts.intersection(set(rhyme_parts))

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
            "rhyme_type": self.rhyme_type,
            "base_tone": self._cached_base_tone,
            "needs_punctuation": self.needs_punctuation,
            "is_finished": self.is_finished,
        }

    @property
    def current_line_text(self) -> str:
        """暴露当前行文本给 LogitsProcessor 做细粒度校验"""
        return self._current_line_text
