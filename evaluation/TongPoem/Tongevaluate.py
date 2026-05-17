#!/usr/bin/env python3
"""
唐诗格律九大规则评估器

基于 BiPro 项目 poem_verifier.py + bipro_poem.py 的完整格律校验逻辑，
对完整的五言/七言绝句/律诗进行逐句逐规则评估。

九大规则:
  1. 字数限制 -- 强制五言或七言，严格禁止六言
  2. 禁止字符 -- 禁止现代白话虚词（的、些、么、了）
  3. 重复检测 -- 禁止句内及跨句 2-gram/3-gram 重复
  4. 未登录字 -- 所有字须在平水韵字典中
  5. 平仄规则 -- 二四六分明，末字收束合律
  6. 孤平检查 -- 平收句检测孤平
  7. 三连同   -- 禁止三连平/三连仄
  8. 押韵一致 -- 基于平水韵一韵到底
  9. 多音字妥协 -- 多音字豁免严格声调检查

参考:
  - bipro/glm/poem_verifier.py (格律校验逻辑)
  - bipro/glm/bipro_poem.py   (状态机调用关系)
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════
#  平水韵词典加载
# ═══════════════════════════════════════════════════════════════

def load_pingshui(path: str = None) -> Tuple[Dict, Dict, List, List]:
    """
    加载平水韵词典。
    返回: (worddict, shengdict, allbu, allsb)
      - worddict:   {char: [韵部编号列表]}
      - shengdict:  {char: [声调列表]}  0=平声, 1=仄声(上/去/入)
      - allbu:      [[每韵部所含字], ...]
      - allsb:      [[平声字列表], [仄声字列表]]
    """
    if path is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', 'bipro', 'glm', 'pingshui.txt'),
            os.path.join('bipro', 'glm', 'pingshui.txt'),
            'pingshui.txt',
        ]
        for p in candidates:
            if os.path.exists(p):
                path = p
                break
        else:
            raise FileNotFoundError("找不到 pingshui.txt，请提供正确路径")

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    bu = 0
    nowsb = 0
    allbu: List[List[str]] = []
    def_chr = ['[', ']', '(', ')', '\n', ' ', '，', '　', '。', '《', '》']
    allsb: List[List[str]] = [[], []]
    worddict: Dict[str, List[int]] = {}
    shengdict: Dict[str, List[int]] = {}

    for line in lines:
        if len(line) < 5:
            continue
        if "其它僻字" in line:
            continue
        words = line.strip()
        if '　' in line:
            bu += 1
            allbu.append([])
            if '平声' in line:
                nowsb = 0
            if '上声' in line:
                nowsb = 1
            if '去声' in line:
                nowsb = 1
            if '入声' in line:
                nowsb = 1
            words = words.split('　')[1]

        currentst1 = 0
        currentst2 = 0
        for num in range(len(words)):
            char = words[num]
            if currentst1 + currentst2 == 0:
                if not (char in def_chr):
                    allbu[-1].append(char)
                    allsb[nowsb].append(char)
                    if char in worddict:
                        if not (bu in worddict[char]):
                            worddict[char].append(bu)
                    else:
                        worddict[char] = [bu]
                    if char in shengdict:
                        if not (nowsb in shengdict[char]):
                            shengdict[char].append(nowsb)
                    else:
                        shengdict[char] = [nowsb]
            if char == '[':
                currentst1 = 1
            if char == ']':
                currentst1 = 0
            if char == '(':
                currentst2 = 1
            if char == ')':
                currentst2 = 0

    return worddict, shengdict, allbu, allsb


# ═══════════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class RuleResult:
    """单条规则的评估结果"""
    rule_id: int
    name: str
    passed: bool
    detail: str = ""
    line_idx: int = -1

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[规则{self.rule_id} {status}] {self.name}: {self.detail}"


@dataclass
class LineEvaluation:
    """单句评估结果"""
    line_idx: int
    line_text: str
    results: List[RuleResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)


@dataclass
class PoemEvaluation:
    """整诗评估结果"""
    poem_type: str = ""
    total_lines: int = 0
    line_evals: List[LineEvaluation] = field(default_factory=list)
    multi_tone_chars: List[Tuple[int, int, str, List[str]]] = field(default_factory=list)
    overall_pass: bool = True
    score: float = 100.0

    @property
    def violations(self) -> List[RuleResult]:
        return [r for le in self.line_evals
                for r in le.results if not r.passed]


# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════

def parse_poem(poem_text: str) -> List[str]:
    """将诗歌文本解析为句子列表（去除标点符号，保留纯汉字）"""
    lines = re.split(r'[，,。.！!？?\n\r]+', poem_text.strip())
    lines = [l.strip() for l in lines if l.strip()]
    return lines


def determine_poem_type(lines: List[str]) -> Tuple[int, int, str]:
    """
    判断诗歌类型。
    返回: (每句字数, 总句数, 类型名)
      - (5, 4, "五言绝句"), (5, 8, "五言律诗"),
      - (7, 4, "七言绝句"), (7, 8, "七言律诗")
    """
    if not lines:
        return 0, 0, "未知"

    lens = [len(l) for l in lines if len(l) in (5, 7)]
    if not lens:
        return 0, 0, "未知"

    most_common = max(set(lens), key=lens.count)
    n_lines = len(lines)

    type_map = {
        (5, 4): "五言绝句",
        (5, 8): "五言律诗",
        (7, 4): "七言绝句",
        (7, 8): "七言律诗",
    }
    type_name = type_map.get((most_common, n_lines),
                             f"{most_common}言{n_lines}句")

    return most_common, n_lines, type_name


# ═══════════════════════════════════════════════════════════════
#  主评估函数 -- 完整复刻 poem_verifier + bipro_poem 状态机
# ═══════════════════════════════════════════════════════════════

def evaluate_poem(poem_text: str,
                  shengdict: Dict = None,
                  wdic: Dict = None,
                  pingshui_path: str = None) -> PoemEvaluation:
    """
    对一首完整的唐诗进行九大规则格律评估。

    复刻原始代码 poem_verifier() + verify_rhy() + bipro_poem.poemgen()
    中的状态机逻辑：

    全局状态变量:
      global_rhy:   首句第 2 字的声调 (0=平, 1=仄, 2=未定)
                   由第 0 句确定，后续不变
      global_endrhy: 首句末字的声调 (0=平收, 1=仄收, 2=未定)
                      由第 0 句确定
      yayun:        韵脚字列表（平声收束的行末字积累于此）
      prev_lines:   前文所有句子（用于重复检测）

    每句处理前，根据行号 ids 设置该句的检验参数：
      ids ∈ [0,3,4,7] -> 期望第 2 字 = global_rhy       (粘)
      ids ∈ [1,2,5,6] -> 期望第 2 字 = 1 - global_rhy  (对)
      ids == 0        -> endrhy = 2 (由本句末字决定)
      ids > 0         -> endrhy = 1 - (ids % 2)
    """
    if shengdict is None or wdic is None:
        wdic, shengdict, _, _ = load_pingshui(pingshui_path)

    evaluation = PoemEvaluation()
    lines = parse_poem(poem_text)

    if not lines:
        evaluation.overall_pass = False
        evaluation.score = 0.0
        return evaluation

    line_len, n_lines, type_name = determine_poem_type(lines)
    evaluation.poem_type = type_name
    evaluation.total_lines = n_lines

    if line_len == 0:
        evaluation.overall_pass = False
        evaluation.score = 0.0
        return evaluation

    min_len = max_len = line_len
    end_tokens = ['。', '，', '！', '？', '.', ',', '!', '?']

    # 全局状态变量
    global_rhy = 2           # 首句第 2 字声调: 0=平, 1=仄, 2=未定
    global_endrhy = 2        # 首句末字声调: 0=平收, 1=仄收, 2=未定
    yayun: List[str] = []    # 韵脚字累积列表
    prev_lines: List[str] = []  # 已处理句子的纯文本

    for ids, line in enumerate(lines):
        line_eval = LineEvaluation(line_idx=ids, line_text=line)

        # -- 规则 1: 字数限制 --
        r1 = _check_length(line, ids, min_len, max_len)
        line_eval.results.append(r1)
        if not r1.passed:
            evaluation.line_evals.append(line_eval)
            evaluation.overall_pass = False
            prev_lines.append(line)
            continue

        # -- 规则 2: 禁止字符 --
        r2 = _check_forbidden_chars(line, ids)
        line_eval.results.append(r2)
        if not r2.passed:
            evaluation.line_evals.append(line_eval)
            evaluation.overall_pass = False
            prev_lines.append(line)
            continue

        # -- 规则 3: 重复检测 --
        prev_text = ''.join(prev_lines)
        r3_list = _check_repetition(line, prev_text, ids)
        for r3 in r3_list:
            line_eval.results.append(r3)
            if not r3.passed:
                evaluation.overall_pass = False

        # -- 规则 4: 未登录字 --
        r4 = _check_unknown_chars(line, shengdict, ids)
        line_eval.results.append(r4)
        if not r4.passed:
            evaluation.overall_pass = False

        if len(line) < 2:
            prev_lines.append(line)
            evaluation.line_evals.append(line_eval)
            continue

        # -- 确定该句的检验用 rhy --
        # 复刻 bipro_poem.py 第 248-252 行的逻辑
        if global_rhy != 2:
            if ids in [1, 2, 5, 6]:
                per_line_rhy = 1 - global_rhy
            else:
                per_line_rhy = global_rhy
        else:
            per_line_rhy = 2

        # -- 确定该句的检验用 endrhy --
        # 复刻 bipro_poem.py 第 380-382 行（refine 中也有类似逻辑）
        if ids > 0:
            per_line_endrhy = 1 - (ids % 2)
        else:
            per_line_endrhy = 2

        # -- 确定该句的 yayun 参数 --
        # 复刻 bipro_poem.py 第 242-245 行
        if ids % 2 == 0:
            verify_yayun: List[str] = []
        else:
            verify_yayun = list(yayun)

        # -- 规则 5: 平仄规则 (二四六分明 + 末字收束) --
        r5, global_rhy = _check_tone_pattern(
            line, shengdict, per_line_rhy, per_line_endrhy, ids, global_rhy)
        line_eval.results.append(r5)
        if not r5.passed:
            evaluation.overall_pass = False

        # -- 规则 6: 孤平检查 --
        # poem_verifier 第 388-407 行: endrhy==0 时才检查
        if per_line_endrhy == 0:
            r6 = _check_guping(line, shengdict, per_line_rhy, ids)
        else:
            r6 = RuleResult(6, "孤平检查", True, "非平收句，无需检查")
        line_eval.results.append(r6)
        if not r6.passed:
            evaluation.overall_pass = False

        # -- 规则 7: 三连同禁止 --
        # 含: 预检查(max_length-3位)、末字不字禁止、三连平/三连仄
        r7 = _check_three_consecutive(
            line, shengdict, per_line_rhy, per_line_endrhy,
            max_len, ids)
        line_eval.results.append(r7)
        if not r7.passed:
            evaluation.overall_pass = False

        # -- 规则 8: 押韵一致 --
        r8 = _check_rhyme(line, wdic, shengdict, verify_yayun,
                          per_line_endrhy, ids)
        line_eval.results.append(r8)
        if not r8.passed:
            evaluation.overall_pass = False

        # -- 规则 9: 多音字妥协 --
        r9, mtc_list = _check_multi_tone(line, shengdict, ids)
        line_eval.results.append(r9)
        for pos, ch, tones in mtc_list:
            evaluation.multi_tone_chars.append((ids, pos, ch, tones))

        # -- 更新全局状态 (复刻 verify_rhy 逻辑) --
        last_char = line[-1]
        pz_last = shengdict.get(last_char, [])

        # 确定该句是否需要押韵（加入 yayun 列表）
        need_yy = 0
        if ids == 0:
            if len(pz_last) == 1 and pz_last[0] == 0:
                need_yy = 1
        if ids % 2 == 1:
            need_yy = 1

        if need_yy == 1:
            yayun.append(last_char)

        # 确定 global_endrhy
        if global_endrhy == 2 and len(pz_last) == 1:
            global_endrhy = pz_last[0]

        # 确定 global_rhy（仅首句未定时）
        if global_rhy == 2:
            if len(line) >= 2:
                pz_2 = shengdict.get(line[1], [])
                if len(pz_2) == 1:
                    global_rhy = pz_2[0]
                    if ids in [1, 2, 5, 6]:
                        global_rhy = 1 - global_rhy
            if global_rhy == 2 and len(line) >= 4:
                pz_4 = shengdict.get(line[3], [])
                if len(pz_4) == 1:
                    global_rhy = 1 - pz_4[0]
                    if ids in [1, 2, 5, 6]:
                        global_rhy = 1 - global_rhy
            if global_rhy == 2 and len(line) >= 6:
                pz_6 = shengdict.get(line[5], [])
                if len(pz_6) == 1:
                    global_rhy = pz_6[0]
                    if ids in [1, 2, 5, 6]:
                        global_rhy = 1 - global_rhy

        prev_lines.append(line)
        evaluation.line_evals.append(line_eval)

    # 计算得分: 基础 100，每条违规扣 10 分
    evaluation.score = max(0.0, 100.0 - len(evaluation.violations) * 10.0)

    return evaluation


# ═══════════════════════════════════════════════════════════════
#  九大规则实现函数
# ═══════════════════════════════════════════════════════════════

def _check_length(line: str, ids: int,
                  min_len: int, max_len: int) -> RuleResult:
    """
    规则 1: 字数限制
    强制五言或七言，严格禁止六言。
    复刻 poem_verifier 第 264-302 行。
    """
    name = "字数限制"
    l = len(line)

    if l == 6:
        return RuleResult(1, name, False,
                          f"第{ids+1}句为六言'{line}'，格律诗禁止六言", ids)
    if l < min_len:
        return RuleResult(1, name, False,
                          f"第{ids+1}句仅{l}字'{line}'，不足{min_len}言", ids)
    if l > max_len:
        return RuleResult(1, name, False,
                          f"第{ids+1}句有{l}字'{line}'，超过{max_len}言", ids)
    return RuleResult(1, name, True,
                      f"第{ids+1}句{l}言，合规", ids)


def _check_forbidden_chars(line: str, ids: int) -> RuleResult:
    """
    规则 2: 禁止字符
    禁止现代白话虚词: 的、些、么、了
    复刻 poem_verifier 第 317-320 行。
    """
    name = "禁止字符"
    forbidden = ['的', '些', '么', '了']
    for ch in line:
        if ch in forbidden:
            return RuleResult(2, name, False,
                              f"第{ids+1}句含现代白话虚词'{ch}'", ids)
    return RuleResult(2, name, True, "", ids)


def _check_repetition(line: str, prev_text: str, ids: int) -> List[RuleResult]:
    """
    规则 3: 重复检测
    - 3-gram 不得出现在前文任何位置（硬性拒绝）
    - 2-gram 不得在句内重复（硬性拒绝）
    - 2-gram 不得在前文出现（硬性拒绝）
    复刻 poem_verifier 第 254-316 行。
    """
    name = "重复检测"
    results = []

    # 3-gram 跨句重复 -- 硬性拒绝 (第 254-258 行)
    for i in range(len(line) - 2):
        trigram = line[i:i + 3]
        if trigram in prev_text:
            return [RuleResult(3, name, False,
                               f"第{ids+1}句 3-gram'{trigram}'与前文重复", ids)]

    # 2-gram 句内 + 跨句重复 -- 硬性拒绝 (第 309-316 行)
    for i in range(len(line) - 1):
        bigram = line[i:i + 2]
        if bigram in line[:i]:
            return [RuleResult(3, name, False,
                               f"第{ids+1}句内 2-gram'{bigram}'重复", ids)]
        if bigram in prev_text:
            return [RuleResult(3, name, False,
                               f"第{ids+1}句 2-gram'{bigram}'与前文重复", ids)]

    results.append(RuleResult(3, name, True, "", ids))
    return results


def _check_unknown_chars(line: str, shengdict: Dict,
                         ids: int) -> RuleResult:
    """
    规则 4: 未登录字
    句中所有汉字必须在平水韵字典中存在。
    复刻 poem_verifier 第 324-328 行。
    """
    name = "未登录字"
    for ch in line:
        if ch not in shengdict:
            return RuleResult(4, name, False,
                              f"第{ids+1}句含未登录字'{ch}'，不在平水韵字典中", ids)
    return RuleResult(4, name, True, "", ids)


def _check_tone_pattern(line: str, shengdict: Dict,
                        per_line_rhy: int, per_line_endrhy: int,
                        ids: int, global_rhy: int) -> Tuple[RuleResult, int]:
    """
    规则 5: 平仄规则
    - 第 2 字平仄须与期望一致 (二)
    - 第 4 字平仄须与第 2 字相反 (四异)
    - 第 6 字（七言）平仄须与第 2 字相同 (六同)
    - 末字声调须符合收束规则 (仄起平收)
    复刻 poem_verifier 第 334-386 行 + 第 427-436 行。
    """
    name = "平仄规则"
    new_global_rhy = global_rhy
    tone_names = {0: "平", 1: "仄"}

    # -- 第 2 字 (index 1) -- 复刻第 334-343 行
    pz1 = shengdict.get(line[1], [])
    if per_line_rhy != 2:
        if len(pz1) == 1:
            if per_line_rhy != pz1[0]:
                return RuleResult(5, name, False,
                                  f"第{ids+1}句第2字'{line[1]}'应为{tone_names[per_line_rhy]}"
                                  f"实为{tone_names[pz1[0]]}", ids), new_global_rhy
    else:
        if len(pz1) == 1:
            new_global_rhy = pz1[0]

    # -- 第 4 字 (index 3) -- 复刻第 348-358 行
    if len(line) >= 4:
        pz2 = shengdict.get(line[3], [])
        if per_line_rhy != 2:
            if len(pz2) == 1:
                if pz2[0] + per_line_rhy != 1:
                    expected = 1 - per_line_rhy
                    return RuleResult(5, name, False,
                                      f"第{ids+1}句第4字'{line[3]}'应为{tone_names[expected]}"
                                      f"(与第2字相反)，实为{tone_names[pz2[0]]}", ids), new_global_rhy
        else:
            if len(pz2) == 1:
                new_global_rhy = 1 - pz2[0]

    # -- 第 6 字 (index 5) -- 仅七言，复刻第 378-386 行
    if len(line) >= 6:
        pz3 = shengdict.get(line[5], [])
        if per_line_rhy != 2:
            if len(pz3) == 1:
                if per_line_rhy != pz3[0]:
                    return RuleResult(5, name, False,
                                      f"第{ids+1}句第6字'{line[5]}'应为{tone_names[per_line_rhy]}"
                                      f"(与第2字相同)，实为{tone_names[pz3[0]]}", ids), new_global_rhy

    # -- 末字收束检查 -- 复刻第 427-436 行
    last_char = line[-1]
    pz_last = shengdict.get(last_char, [])
    if per_line_endrhy != 2:
        if len(pz_last) == 1:
            if per_line_endrhy != pz_last[0]:
                return RuleResult(5, name, False,
                                  f"第{ids+1}句末字'{last_char}'应为{tone_names[per_line_endrhy]}收"
                                  f"，实为{tone_names[pz_last[0]]}", ids), new_global_rhy

    return RuleResult(5, name, True, "", ids), new_global_rhy


def _check_guping(line: str, shengdict: Dict,
                  per_line_rhy: int, ids: int) -> RuleResult:
    """
    规则 6: 孤平检查
    仅对平收句 (endrhy==0) 进行检查:
    - 平起式 (rhy=0): 第 1、3 字不可全仄，否则第 2 字平声孤立
    - 仄起式 (rhy=1): 第 3、5 字不可全仄，否则第 4 字平声孤立
    复刻 poem_verifier 第 388-407 行。
    """
    name = "孤平检查"

    if per_line_rhy == 0:
        if len(line) >= 3:
            pz1 = shengdict.get(line[0], [])
            pz3 = shengdict.get(line[2], [])
            if len(pz1) + len(pz3) == 2:
                if pz1[0] + pz3[0] == 2:
                    return RuleResult(6, name, False,
                                      f"第{ids+1}句孤平: 第1字'{line[0]}'第3字'{line[2]}'皆仄，"
                                      f"第2字'{line[1]}'平声孤立", ids)

    if per_line_rhy == 1:
        if len(line) >= 5:
            pz1 = shengdict.get(line[2], [])
            pz3 = shengdict.get(line[4], [])
            if len(pz1) + len(pz3) == 2:
                if pz1[0] + pz3[0] == 2:
                    return RuleResult(6, name, False,
                                      f"第{ids+1}句孤平: 第3字'{line[2]}'第5字'{line[4]}'皆仄，"
                                      f"第4字'{line[3]}'平声孤立", ids)

    return RuleResult(6, name, True, "", ids)


def _check_three_consecutive(line: str, shengdict: Dict,
                              per_line_rhy: int, per_line_endrhy: int,
                              max_len: int, ids: int) -> RuleResult:
    """
    规则 7: 三连同禁止
    - 预检查: 倒数第 (max_length-3) 位声调不得与收束声调连续相同
    - 末字不得为'不'
    - 末三字不得三连平 (全部平声)
    - 末三字不得三连仄 (全部仄声)
    复刻 poem_verifier 第 363-376 行 + 第 411-425 行。
    """
    name = "三连同"

    # -- 预检查: 倒数第三位 -- 复刻第 363-376 行
    if len(line) > max_len - 3:
        if per_line_endrhy != 2:
            wrhy = per_line_rhy
            if max_len == 5:
                wrhy = 1 - per_line_rhy
            if per_line_endrhy == wrhy:
                pz = shengdict.get(line[max_len - 3], [])
                if len(pz) == 1:
                    if pz[0] == wrhy:
                        tone_name = "平" if wrhy == 0 else "仄"
                        return RuleResult(7, name, False,
                                          f"第{ids+1}句三连同预检: 倒数第3位'{line[max_len-3]}'"
                                          f"为{tone_name}声，与收束声调连续", ids)

    # -- 末字检查 -- 复刻第 412-413 行
    if line[-1] == '不':
        return RuleResult(7, name, False,
                          f"第{ids+1}句以'不'字结尾，禁止", ids)

    # -- 末三字三连同检查 -- 复刻第 414-425 行
    if len(line) >= 3:
        pz_3 = shengdict.get(line[-3], [])
        pz_2 = shengdict.get(line[-2], [])
        pz_1 = shengdict.get(line[-1], [])

        if len(pz_1) + len(pz_2) + len(pz_3) == 3:
            tone_sum = pz_1[0] + pz_2[0] + pz_3[0]
            if tone_sum == 0:
                return RuleResult(7, name, False,
                                  f"第{ids+1}句三连平: '{line[-3:]}'", ids)
            if tone_sum == 3:
                return RuleResult(7, name, False,
                                  f"第{ids+1}句三连仄: '{line[-3:]}'", ids)

    return RuleResult(7, name, True, "", ids)


def _check_rhyme(line: str, wdic: Dict, shengdict: Dict,
                 verify_yayun: List[str], per_line_endrhy: int,
                 ids: int) -> RuleResult:
    """
    规则 8: 押韵一致
    平收句 (endrhy==0) 中，若 yayun 列表非空，则末字须与 yayun
    中的韵脚字同属一个平水韵韵部（一韵到底）。
    复刻 poem_verifier 第 438-455 行。
    """
    name = "押韵一致"

    if len(verify_yayun) == 0:
        return RuleResult(8, name, True, "韵脚未定，暂不检查", ids)

    if per_line_endrhy != 0:
        return RuleResult(8, name, True, "仄收句，无需押韵", ids)

    last_char = line[-1]
    final1 = wdic.get(last_char, [])
    if not final1:
        return RuleResult(8, name, True, "", ids)

    final2 = []
    for yc in verify_yayun:
        f2 = wdic.get(yc, [])
        if f2:
            final2.append(f2)

    if not final2:
        return RuleResult(8, name, True, "", ids)

    matched = False
    for f1 in final1:
        for td in final2:
            if f1 in td:
                matched = True
                break
        if matched:
            break

    if not matched:
        yayun_str = ''.join(verify_yayun)
        return RuleResult(8, name, False,
                          f"第{ids+1}句末字'{last_char}'不押韵"
                          f"(韵脚: {yayun_str})", ids)

    return RuleResult(8, name, True, "", ids)


def _check_multi_tone(line: str, shengdict: Dict,
                      ids: int) -> Tuple[RuleResult, List[Tuple[int, str, List[str]]]]:
    """
    规则 9: 多音字妥协
    标记句中的多音字 (shengdict[char] 长度 > 1)。
    这些字在平仄检查中因声调不确定而被豁免。
    本规则始终通过，仅作信息标记。
    """
    name = "多音字妥协"
    mtc_list: List[Tuple[int, str, List[str]]] = []
    for i, ch in enumerate(line):
        if ch in shengdict and len(shengdict[ch]) > 1:
            tone_labels = ['平' if t == 0 else '仄' for t in shengdict[ch]]
            mtc_list.append((i, ch, tone_labels))

    if mtc_list:
        chars_str = ', '.join([f"'{c}'({'/'.join(t)})" for _, c, t in mtc_list])
        detail = f"第{ids+1}句多音字: {chars_str}"
    else:
        detail = ""
    return RuleResult(9, name, True, detail, ids), mtc_list


# ═══════════════════════════════════════════════════════════════
#  便捷接口
# ═══════════════════════════════════════════════════════════════

def score_poem(poem_text: str, pingshui_path: str = None) -> float:
    """快速评分 (0-100)，基础分 100，每条违规扣 10 分"""
    result = evaluate_poem(poem_text, pingshui_path=pingshui_path)
    return result.score


def quick_check(poem_text: str, pingshui_path: str = None) -> bool:
    """快速检查是否通过所有九大规则"""
    result = evaluate_poem(poem_text, pingshui_path=pingshui_path)
    return result.overall_pass


def print_evaluation(poem_text: str, pingshui_path: str = None):
    """格式化打印完整评估结果"""
    result = evaluate_poem(poem_text, pingshui_path=pingshui_path)

    print("=" * 56)
    print(f"  唐诗格律九大规则评估 -- {result.poem_type}")
    print("=" * 56)

    for le in result.line_evals:
        print(f"\n{'-' * 40}")
        print(f"  第 {le.line_idx + 1} 句: {le.line_text}")
        print(f"{'-' * 40}")
        for r in le.results:
            status = "[PASS]" if r.passed else "[FAIL]"
            line = f"  {status} 规则{r.rule_id} {r.name}"
            if r.detail:
                line += f" -- {r.detail}"
            print(line)

    print(f"\n{'=' * 56}")
    if result.overall_pass:
        print("  整体结果: 全部通过 [PASS]")
    else:
        print(f"  整体结果: 存在违规 [FAIL] ({len(result.violations)} 条)")
        for v in result.violations:
            print(f"    -> {v}")
    print(f"  综合得分: {result.score}/100")

    if result.multi_tone_chars:
        items = [f"第{ids+1}句'{c}'({'/'.join(t)})"
                 for ids, _, c, t in result.multi_tone_chars]
        print(f"  多音字豁免: {', '.join(items)}")
    print("=" * 56)


# ═══════════════════════════════════════════════════════════════
#  命令行入口
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    test_cases = [
        # 李白《静夜思》(五言绝句 -- 平起首句入韵)
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
        # 王之涣《登鹳雀楼》(五言绝句 -- 仄起仄收)
        "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
        # 李白《早发白帝城》(七言绝句)
        "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
        # 杜甫《春望》(五言律诗)
        "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。"
        "烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。",
    ]

    if len(sys.argv) > 1:
        if sys.argv[1] == '--file' and len(sys.argv) > 2:
            with open(sys.argv[2], 'r', encoding='utf-8') as f:
                poem = f.read()
        else:
            poem = sys.argv[1]
        print_evaluation(poem)
    else:
        print("用法: python evaluate.py '<完整诗歌>'")
        print("     python evaluate.py --file <诗歌文件>\n")
        print("运行内置测试用例...\n")
        for i, test_poem in enumerate(test_cases):
            print(f"\n{'#' * 56}")
            print(f"  测试用例 {i + 1}")
            print(f"{'#' * 56}")
            print_evaluation(test_poem)
