#!/usr/bin/env python3
"""
Classical Chinese Poetry Evaluator — unified entry for Songci (宋词) and Tang Poem (唐诗).

Supports two evaluation backends sharing the same 3-component scoring system
(Structure 40% + Tonal 30% + Rhyme 30%) based on 中华新韵 (Xinyun) via pypinyin.

Input layout:
    evaluate_input/
        Songci/          ← one .txt per 词牌 (or one .json)
        Tongpoem/        ← one .txt per poem type

Output layout:
    evaluation_output/
        Songci/          ← *_evaluation.json
        Tongpoem/        ← *_evaluation.json
"""

import json
import os
import re
import sys
import io
from collections import Counter
from pypinyin import pinyin, Style
from tqdm import tqdm

# ============================================================
#  Shared utilities — pypinyin-based tone & rhyme (中华新韵)
# ============================================================

_XINYUN_FINAL_MAP = {
    'a': '一麻', 'ia': '一麻', 'ua': '一麻',
    'o': '二波', 'e': '二波', 'uo': '二波',
    'ie': '三皆', 'ue': '三皆',
    'ai': '四开', 'uai': '四开',
    'ei': '五微', 'ui': '五微', 'uei': '五微',
    'ao': '六豪', 'iao': '六豪',
    'ou': '七尤', 'iu': '七尤', 'iou': '七尤',
    'an': '八寒', 'ian': '八寒', 'uan': '八寒', 'van': '八寒',
    'en': '九文', 'in': '九文', 'un': '九文', 'uen': '九文', 'vn': '九文',
    'ang': '十唐', 'iang': '十唐', 'uang': '十唐',
    'eng': '十一庚', 'ing': '十一庚', 'ong': '十一庚',
    'iong': '十一庚', 'ueng': '十一庚',
    'er': '十二齐',
    'v': '十二齐',
    've': '三皆',
}

_ZHI_INITIALS = {'zh', 'ch', 'sh', 'r', 'z', 'c', 's'}
_JU_INITIALS = {'j', 'q', 'x', 'y'}


def get_char_tone(char):
    """Return 平/仄/中 for a single character using pypinyin."""
    try:
        tone_char = pinyin(char, style=Style.TONE3, heteronym=False)[0][0][-1]
        if tone_char in '12':
            return '平'
        elif tone_char in '34':
            return '仄'
        return '中'
    except (IndexError, TypeError):
        return '中'


def get_char_rhyme(char):
    """Return frozenset of 中华新韵 category names for a character."""
    try:
        all_finals = pinyin(char, style=Style.FINALS, heteronym=True)[0]
        all_initials = pinyin(char, style=Style.INITIALS, heteronym=True)[0]
        if not all_finals:
            return frozenset({'none'})
    except (IndexError, TypeError):
        return frozenset({'none'})

    categories = set()
    for i, final in enumerate(all_finals):
        initial = all_initials[i] if i < len(all_initials) else (all_initials[0] if all_initials else '')

        if not final:
            continue

        if final == 'i':
            if initial in _ZHI_INITIALS:
                categories.add('十三支')
            else:
                categories.add('十二齐')
        elif final == 'u':
            if initial in _JU_INITIALS:
                categories.add('十二齐')
            else:
                categories.add('十四姑')
        else:
            category = _XINYUN_FINAL_MAP.get(final)
            if category:
                categories.add(category)

    return frozenset(categories) if categories else frozenset({'none'})


def _parse_poem_lines(text):
    """Split poem text into lines by Chinese punctuation."""
    lines = re.split(r'[，,。.！!？?、；;\n\r]+', text.strip())
    return [line.strip() for line in lines if line.strip()]


def extract_poem_text(raw_output):
    """Extract poem body after the [content] marker."""
    m = re.search(r'\[content\]\s*(.+)', raw_output, re.DOTALL)
    if m:
        text = m.group(1).strip()
        return text if text else None
    return None


# ============================================================
#  SongciEvaluator (宋词)
# ============================================================

class SongciEvaluator:
    """Evaluate generated Songci against meter rules from songci.json."""

    def __init__(self, meter_path):
        with open(meter_path, 'r', encoding='utf-8') as f:
            self.meters = json.load(f)

    def _build_total_view(self, meter_entry):
        chars_per_line = []
        rhyme_groups = []
        tonal_patterns = []
        expanded_offset = 0

        for stanza_key in ['stanza1', 'stanza2']:
            stanza = meter_entry.get(stanza_key)
            if not stanza:
                continue

            stanza_lines = stanza.get('lines', [])
            line_to_subclauses = []

            for tonal_line in stanza_lines:
                sub_clauses = tonal_line.split('、')
                sub_indices = []
                for sub in sub_clauses:
                    cleaned = re.sub(r'[/\s]', '', sub)
                    if cleaned:
                        chars_per_line.append(len(cleaned))
                        tonal_patterns.append(cleaned)
                        sub_indices.append(expanded_offset)
                        expanded_offset += 1
                line_to_subclauses.append(sub_indices)

            for key in sorted(stanza):
                if key.startswith('rhyme_') and key.endswith('_positions'):
                    group_positions = []
                    for pos in stanza[key]:
                        line_idx = pos - 1
                        if line_idx < len(line_to_subclauses):
                            sub_indices = line_to_subclauses[line_idx]
                            if sub_indices:
                                group_positions.append(sub_indices[-1] + 1)
                    if group_positions:
                        rhyme_groups.append(group_positions)

        return {
            'chars_per_line': chars_per_line,
            'rhyme_groups': rhyme_groups,
            'tonal_patterns': tonal_patterns,
        }

    def _get_template_line_count(self, meter_entry):
        count = 0
        for stanza_key in ['stanza1', 'stanza2']:
            stanza = meter_entry.get(stanza_key)
            if stanza:
                for tonal_line in stanza.get('lines', []):
                    count += len([c for c in tonal_line.split('、') if re.sub(r'[/\s]', '', c)])
        return count

    def _calculate_score(self, generated_lines, meter_entry):
        scores = {"structure": 0.0, "tonal": 0.0, "rhyme": 0.0}
        details = {"structure": {}, "tonal": {}, "rhyme": {}}
        total = self._build_total_view(meter_entry)

        template_chars = total['chars_per_line']
        tonal_patterns = total['tonal_patterns']
        rhyme_groups = total['rhyme_groups']

        if not template_chars:
            return scores, details

        n_template = len(template_chars)
        n_generated = len(generated_lines)
        n_common = min(n_template, n_generated)

        # --- 1. Structure Score (40%) ---
        if n_template > 0:
            mismatches = []
            for i in range(n_common):
                actual_len = len(generated_lines[i])
                expected_len = template_chars[i]
                if actual_len != expected_len:
                    mismatches.append({
                        "clause_index": i,
                        "clause": generated_lines[i],
                        "actual_len": actual_len,
                        "expected_len": expected_len,
                    })
            correct = n_common - len(mismatches)
            scores['structure'] = correct / n_template
            details['structure'] = {
                "mismatches": mismatches,
                "missing_template_clauses": n_template - n_common if n_template > n_generated else 0,
                "extra_generated_clauses": n_generated - n_common if n_generated > n_template else 0,
            }

        # --- 2. Tonal Score (30%) ---
        tonal_mismatches = []
        skipped_clauses = []
        total_chars, matching_chars = 0, 0
        if tonal_patterns:
            for i in range(n_common):
                if i >= len(tonal_patterns):
                    break
                pattern = tonal_patterns[i]
                clause = generated_lines[i]
                if len(clause) != len(pattern):
                    skipped_clauses.append({
                        "clause_index": i,
                        "clause": clause,
                        "reason": f"length mismatch (clause:{len(clause)} vs pattern:{len(pattern)})",
                    })
                    continue
                clause_mismatches = []
                for j, char in enumerate(clause):
                    required = pattern[j]
                    actual = get_char_tone(char)
                    total_chars += 1
                    if required == '中' or actual == required:
                        matching_chars += 1
                    else:
                        clause_mismatches.append({
                            "char": char,
                            "position_in_clause": j,
                            "required_tone": required,
                            "actual_tone": actual,
                        })
                if clause_mismatches:
                    tonal_mismatches.append({
                        "clause_index": i,
                        "clause": clause,
                        "pattern": pattern,
                        "mismatches": clause_mismatches,
                        "mismatch_count": len(clause_mismatches),
                        "clause_total_chars": len(clause),
                    })
            if total_chars > 0:
                scores['tonal'] = matching_chars / total_chars
            details['tonal'] = {
                "mismatches": tonal_mismatches,
                "skipped_clauses": skipped_clauses,
                "total_match_count": matching_chars,
                "total_char_count": total_chars,
            }

        # --- 3. Rhyme Score (30%) ---
        if rhyme_groups:
            group_results = []
            for group in rhyme_groups:
                chars = []
                char_category_sets = []
                group_xinyun_map = {}
                for pos in group:
                    idx = pos - 1
                    if idx < n_generated and generated_lines[idx]:
                        char = generated_lines[idx][-1]
                        chars.append(char)
                        cat_set = get_char_rhyme(char)
                        char_category_sets.append(cat_set)
                        for cat in cat_set:
                            if cat != 'none':
                                group_xinyun_map.setdefault(cat, []).append({
                                    "position": pos,
                                    "char": char,
                                })

                if not chars:
                    continue

                category_counts = Counter()
                for cat_set in char_category_sets:
                    for cat in cat_set:
                        if cat != 'none':
                            category_counts[cat] += 1

                if category_counts:
                    dominant = category_counts.most_common(1)[0]
                    group_score = dominant[1] / len(chars)
                else:
                    dominant = (None, 0)
                    group_score = 0.0

                group_results.append({
                    "positions": group,
                    "chars": chars,
                    "xinyun_map": group_xinyun_map,
                    "dominant_xinyun": dominant[0],
                    "match_count": dominant[1],
                    "total_count": len(chars),
                    "group_score": round(group_score, 4),
                })

            if group_results:
                scores['rhyme'] = sum(g["group_score"] for g in group_results) / len(group_results)
                details['rhyme'] = {
                    "groups": group_results,
                    "num_groups": len(group_results),
                    "average_group_score": round(scores['rhyme'], 4),
                }
            else:
                details['rhyme'] = {
                    "groups": [],
                    "num_groups": 0,
                    "average_group_score": 0.0,
                }

        return scores, details

    def evaluate(self, cipai, output):
        generated_lines = _parse_poem_lines(output)
        meter_entry = self.meters.get(cipai)

        if not meter_entry:
            return {
                "error": f"Cipai '{cipai}' not found in meter data.",
                "total_score_percentage": 0.0,
            }

        component_scores, score_details = self._calculate_score(generated_lines, meter_entry)
        weights = {"S": 0.4, "T": 0.3, "R": 0.3}
        total_score = (
            weights["S"] * component_scores["structure"]
            + weights["T"] * component_scores["tonal"]
            + weights["R"] * component_scores["rhyme"]
        )

        return {
            "cipai": cipai,
            "variant": meter_entry.get("variant", ""),
            "rhyme_type": meter_entry.get("rhyme_type", ""),
            "length_category": meter_entry.get("length_category", ""),
            "total_score_percentage": round(total_score * 100, 2),
            "structure_score_percentage": round(component_scores["structure"] * 100, 2),
            "tonal_score_percentage": round(component_scores["tonal"] * 100, 2),
            "rhyme_score_percentage": round(component_scores["rhyme"] * 100, 2),
            "parsed_clauses": generated_lines,
            "parsed_clause_count": len(generated_lines),
            "template_line_count": self._get_template_line_count(meter_entry),
            "scoring_details": {
                "structure": score_details.get("structure", {}),
                "tonal": score_details.get("tonal", {}),
                "rhyme": score_details.get("rhyme", {}),
            },
        }


# ============================================================
#  TangPoemEvaluator (唐诗)
# ============================================================

class TangPoemEvaluator:
    """Evaluate Tang poems using the same 3-component scoring as Songci.

    Standard regulated-verse (律诗/绝句) tonal patterns are built-in.
    Uses pypinyin + 中华新韵 for tone and rhyme detection.
    """

    # ---- standard tonal patterns (一三五不论 applied: odd positions → 中) ----

    _WUYAN = {
        'A': '中仄中平仄',          # 仄仄平平仄
        'B': '中平中仄平',          # 平平仄仄平
        'C': '中平中仄仄',          # 平平平仄仄
        'D': '中仄仄平平',          # 仄仄仄平平
    }

    _QIYAN = {
        'A': '中平中仄中平仄',      # 平平仄仄平平仄
        'B': '中仄中平中仄平',      # 仄仄平平仄仄平
        'C': '中仄中平中仄仄',      # 仄仄平平平仄仄
        'D': '中平中仄仄平平',      # 平平仄仄仄平平
    }

    # arrangement tables — keys are (起式, 首句入韵)
    _ARRANGEMENT_4 = {   # 绝句
        ('仄', False): ['A', 'B', 'C', 'D'],
        ('仄', True):  ['D', 'B', 'C', 'D'],
        ('平', False): ['C', 'D', 'A', 'B'],
        ('平', True):  ['B', 'D', 'A', 'B'],
    }

    _ARRANGEMENT_8 = {   # 律诗
        ('仄', False): ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],
        ('仄', True):  ['D', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],
        ('平', False): ['C', 'D', 'A', 'B', 'C', 'D', 'A', 'B'],
        ('平', True):  ['B', 'D', 'A', 'B', 'C', 'D', 'A', 'B'],
    }

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    #  Detection helpers
    # ------------------------------------------------------------------

    def _determine_poem_type(self, lines):
        """Return (chars_per_line, num_lines, type_name) or (0, 0, 'unknown')."""
        if not lines:
            return 0, 0, "未知"

        lens = [len(l) for l in lines if len(l) in (5, 7)]
        if not lens:
            return 0, 0, "未知"

        char_count = max(set(lens), key=lens.count)
        n_lines = len(lines)

        type_map = {
            (5, 4): "五言绝句",
            (5, 8): "五言律诗",
            (7, 4): "七言绝句",
            (7, 8): "七言律诗",
        }
        type_name = type_map.get((char_count, n_lines),
                                 f"{char_count}言{n_lines}句")
        return char_count, n_lines, type_name

    def _detect_pattern_key(self, lines):
        """Detect 平起/仄起 and 首句入韵 from the first line.

        Returns ('平', True/False), ('仄', True/False), or falls back to ('仄', False).
        """
        if not lines or len(lines[0]) < 2:
            return ('仄', False)

        first_line = lines[0]
        tone_2nd = get_char_tone(first_line[1])    # 2nd char → 平起/仄起
        tone_last = get_char_tone(first_line[-1])  # last char → 入韵?

        qishi = '平' if tone_2nd == '平' else '仄'  # 中 → treat as 仄
        ruyun = (tone_last == '平')

        return (qishi, ruyun)

    # ------------------------------------------------------------------
    #  Template generation
    # ------------------------------------------------------------------

    def _generate_template(self, char_count, num_lines, pattern_key):
        """Generate a list of tonal-pattern strings for every line.

        Returns list of strings like ['中仄中平仄', '中平中仄平', ...].
        """
        if pattern_key is None:
            return None

        if char_count == 5:
            base = self._WUYAN
        elif char_count == 7:
            base = self._QIYAN
        else:
            return None

        if num_lines == 4:
            arrangement = self._ARRANGEMENT_4.get(pattern_key)
        elif num_lines == 8:
            arrangement = self._ARRANGEMENT_8.get(pattern_key)
        else:
            return None

        if arrangement is None:
            return None

        return [base[k] for k in arrangement]

    # ------------------------------------------------------------------
    #  Main evaluate
    # ------------------------------------------------------------------

    def evaluate(self, poem_text):
        lines = _parse_poem_lines(poem_text)

        char_count, num_lines, type_name = self._determine_poem_type(lines)

        if char_count == 0:
            return {
                "error": f"Cannot determine poem type from text.",
                "poem_type": type_name,
                "total_score_percentage": 0.0,
            }

        pattern_key = self._detect_pattern_key(lines)
        tonal_template = self._generate_template(char_count, num_lines, pattern_key)

        n_generated = len(lines)

        # --- Structure Score (40%) ---
        mismatches = []
        for i, line in enumerate(lines):
            if len(line) != char_count:
                mismatches.append({
                    "line_index": i,
                    "line": line,
                    "actual_len": len(line),
                    "expected_len": char_count,
                })

        structure_score = (n_generated - len(mismatches)) / max(n_generated, 1)
        # Penalise wrong total line count
        if num_lines and n_generated != num_lines:
            structure_score = min(structure_score, 0.5)

        # --- Tonal Score (30%) ---
        tonal_mismatches = []
        total_tonal_chars = 0
        matching_tonal_chars = 0

        if tonal_template:
            n_eval = min(n_generated, len(tonal_template))
            for i in range(n_eval):
                line = lines[i]
                pattern = tonal_template[i]
                if len(line) != len(pattern):
                    continue
                line_mismatches = []
                for j, char in enumerate(line):
                    required = pattern[j]
                    actual = get_char_tone(char)
                    total_tonal_chars += 1
                    if required == '中' or actual == required:
                        matching_tonal_chars += 1
                    else:
                        line_mismatches.append({
                            "char": char,
                            "position_in_line": j,
                            "required_tone": required,
                            "actual_tone": actual,
                        })
                if line_mismatches:
                    tonal_mismatches.append({
                        "line_index": i,
                        "line": line,
                        "pattern": pattern,
                        "mismatches": line_mismatches,
                        "mismatch_count": len(line_mismatches),
                    })

        tonal_score = matching_tonal_chars / max(total_tonal_chars, 1)

        # --- Rhyme Score (30%) ---
        # Tang poem rhyme rule: even-numbered lines (1-indexed: 2,4,6,8) must rhyme.
        # First line may optionally enter the rhyme if it is 平收.
        rhyme_positions = []
        for i in range(num_lines):
            line_idx = i  # 0-indexed
            if line_idx % 2 == 1:  # even lines in 1-indexed
                rhyme_positions.append(line_idx + 1)  # 1-indexed
        # optionally include first line
        if lines and len(lines[0]) > 0 and get_char_tone(lines[0][-1]) == '平':
            rhyme_positions.insert(0, 1)

        rhyme_chars = []
        rhyme_category_sets = []
        rhyme_xinyun_map = {}
        for pos in rhyme_positions:
            idx = pos - 1
            if idx < n_generated and lines[idx]:
                char = lines[idx][-1]
                rhyme_chars.append(char)
                cat_set = get_char_rhyme(char)
                rhyme_category_sets.append(cat_set)
                for cat in cat_set:
                    if cat != 'none':
                        rhyme_xinyun_map.setdefault(cat, []).append({
                            "position": pos,
                            "char": char,
                        })

        if rhyme_chars:
            category_counts = Counter()
            for cat_set in rhyme_category_sets:
                for cat in cat_set:
                    if cat != 'none':
                        category_counts[cat] += 1

            if category_counts:
                dominant = category_counts.most_common(1)[0]
                rhyme_score = dominant[1] / len(rhyme_chars)
            else:
                dominant = (None, 0)
                rhyme_score = 0.0

            rhyme_details = {
                "rhyme_positions": rhyme_positions,
                "chars": rhyme_chars,
                "dominant_xinyun": dominant[0],
                "match_count": dominant[1],
                "total_count": len(rhyme_chars),
                "group_score": round(rhyme_score, 4),
            }
        else:
            rhyme_score = 0.0
            rhyme_details = {"rhyme_positions": [], "group_score": 0.0}

        # --- Final aggregation ---
        weights = {"S": 0.4, "T": 0.3, "R": 0.3}
        total_score = (
            weights["S"] * structure_score
            + weights["T"] * tonal_score
            + weights["R"] * rhyme_score
        )

        return {
            "poem_type": type_name,
            "char_count": char_count,
            "detected_lines": num_lines,
            "actual_lines": n_generated,
            "detected_pattern": f"{pattern_key[0]}起{'首句入韵' if pattern_key and pattern_key[1] else '首句不入韵'}" if pattern_key else "未知",
            "total_score_percentage": round(total_score * 100, 2),
            "structure_score_percentage": round(structure_score * 100, 2),
            "tonal_score_percentage": round(tonal_score * 100, 2),
            "rhyme_score_percentage": round(rhyme_score * 100, 2),
            "parsed_lines": lines,
            "parsed_line_count": n_generated,
            "tonal_template": tonal_template,
            "scoring_details": {
                "structure": {
                    "mismatches": mismatches,
                    "expected_chars_per_line": char_count,
                    "expected_lines": num_lines,
                },
                "tonal": {
                    "mismatches": tonal_mismatches,
                    "total_match_count": matching_tonal_chars,
                    "total_char_count": total_tonal_chars,
                },
                "rhyme": rhyme_details,
            },
        }


# ============================================================
#  Batch evaluation helpers
# ============================================================

def _collect_input_files(input_dir):
    """Return list of .txt file paths under input_dir (recursive)."""
    files = []
    if not os.path.isdir(input_dir):
        return files
    for root, _dirs, filenames in os.walk(input_dir):
        for fn in filenames:
            if fn.endswith('.txt'):
                files.append(os.path.join(root, fn))
    return files


def _process_songci_file(filepath, evaluator, output_dir):
    """Evaluate a single Songci input file (may contain multiple works)."""
    filename = os.path.basename(filepath)
    cipai = re.sub(r'[\._\-·].*', '', filename).strip()

    if cipai not in evaluator.meters:
        print(f"  [SKIP] '{cipai}' (from '{filename}') not in meter data")
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    works = re.split(r'===+\s*作品\s*\d*\s*===+', raw_text)
    if len(works) <= 1:
        works = [raw_text]

    poem_results = []
    total_sum = structure_sum = tonal_sum = rhyme_sum = 0.0
    poem_count = 0

    for i, work in enumerate(works):
        work = work.strip()
        if not work:
            continue
        poem_text = extract_poem_text(work)
        if not poem_text:
            continue

        ev = evaluator.evaluate(cipai, poem_text)
        poem_results.append({
            "work_index": i,
            "poem_text": poem_text,
            "evaluation": ev,
        })

        if "error" not in ev:
            total_sum += ev.get("total_score_percentage", 0)
            structure_sum += ev.get("structure_score_percentage", 0)
            tonal_sum += ev.get("tonal_score_percentage", 0)
            rhyme_sum += ev.get("rhyme_score_percentage", 0)
            poem_count += 1

    avg_scores = {}
    if poem_count > 0:
        avg_scores = {
            "average_total_score": round(total_sum / poem_count, 2),
            "average_structure_score": round(structure_sum / poem_count, 2),
            "average_tonal_score": round(tonal_sum / poem_count, 2),
            "average_rhyme_score": round(rhyme_sum / poem_count, 2),
        }

    out_name = filename.replace('.txt', '_evaluation.json')
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "source_file": filename,
            "cipai": cipai,
            "works_evaluated": poem_count,
            "average_scores": avg_scores,
            "individual_results": poem_results,
        }, f, ensure_ascii=False, indent=2)

    return {
        "source_file": filename,
        "cipai": cipai,
        "works_evaluated": poem_count,
        "average_scores": avg_scores,
    }


def _process_tongpoem_file(filepath, evaluator, output_dir):
    """Evaluate a single Tongpoem input file (may contain multiple works)."""
    filename = os.path.basename(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    works = re.split(r'===+\s*作品\s*\d*\s*===+', raw_text)
    if len(works) <= 1:
        works = [raw_text]

    poem_results = []
    total_sum = structure_sum = tonal_sum = rhyme_sum = 0.0
    poem_count = 0

    for i, work in enumerate(works):
        work = work.strip()
        if not work:
            continue
        poem_text = extract_poem_text(work)
        if not poem_text:
            continue

        ev = evaluator.evaluate(poem_text)
        poem_results.append({
            "work_index": i,
            "poem_text": poem_text,
            "evaluation": ev,
        })

        if "error" not in ev:
            total_sum += ev.get("total_score_percentage", 0)
            structure_sum += ev.get("structure_score_percentage", 0)
            tonal_sum += ev.get("tonal_score_percentage", 0)
            rhyme_sum += ev.get("rhyme_score_percentage", 0)
            poem_count += 1

    avg_scores = {}
    if poem_count > 0:
        avg_scores = {
            "average_total_score": round(total_sum / poem_count, 2),
            "average_structure_score": round(structure_sum / poem_count, 2),
            "average_tonal_score": round(tonal_sum / poem_count, 2),
            "average_rhyme_score": round(rhyme_sum / poem_count, 2),
        }

    out_name = filename.replace('.txt', '_evaluation.json')
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "source_file": filename,
            "poem_type": poem_results[0]["evaluation"].get("poem_type", "") if poem_results else "",
            "works_evaluated": poem_count,
            "average_scores": avg_scores,
            "individual_results": poem_results,
        }, f, ensure_ascii=False, indent=2)

    return {
        "source_file": filename,
        "poem_type": poem_results[0]["evaluation"].get("poem_type", "") if poem_results else "",
        "works_evaluated": poem_count,
        "average_scores": avg_scores,
    }


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    # Fix Windows console encoding
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    import argparse

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Evaluate Songci + Tang poems against tonal/structure/rhyme rules")
    parser.add_argument("--meter", default=os.path.join(SCRIPT_DIR, "..", "Meter", "songci.json"),
                        help="Path to Songci meter JSON")
    parser.add_argument("--input", default=os.path.join(SCRIPT_DIR, "evaluate_input"),
                        help="Root input directory (expects Songci/ and Tongpoem/ subdirs)")
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "evaluation_output"),
                        help="Root output directory")
    args = parser.parse_args()

    METER_FILE = args.meter
    INPUT_ROOT = args.input
    OUTPUT_ROOT = args.output

    # --- Songci ---
    songci_input_dir = os.path.join(INPUT_ROOT, "Songci")
    songci_output_dir = os.path.join(OUTPUT_ROOT, "Songci")
    os.makedirs(songci_output_dir, exist_ok=True)

    songci_evaluator = SongciEvaluator(METER_FILE) if os.path.exists(METER_FILE) else None
    if songci_evaluator:
        print(f"Loaded {len(songci_evaluator.meters)} cipai from '{METER_FILE}'")
    else:
        print(f"[WARN] Meter file not found: {METER_FILE}")

    songci_summary = []
    if songci_evaluator and os.path.isdir(songci_input_dir):
        songci_files = _collect_input_files(songci_input_dir)
        print(f"\n{'='*56}")
        print(f"  Songci evaluation — {len(songci_files)} file(s)")
        print(f"{'='*56}")
        for fp in tqdm(songci_files, desc="Evaluating Songci"):
            res = _process_songci_file(fp, songci_evaluator, songci_output_dir)
            if res:
                songci_summary.append(res)
    else:
        print(f"[SKIP] Songci: no evaluator or input dir missing")

    # --- Tongpoem ---
    tongpoem_input_dir = os.path.join(INPUT_ROOT, "Tongpoem")
    tongpoem_output_dir = os.path.join(OUTPUT_ROOT, "Tongpoem")
    os.makedirs(tongpoem_output_dir, exist_ok=True)

    tongpoem_evaluator = TangPoemEvaluator()
    tongpoem_summary = []

    if os.path.isdir(tongpoem_input_dir):
        tongpoem_files = _collect_input_files(tongpoem_input_dir)
        print(f"\n{'='*56}")
        print(f"  Tang poem evaluation — {len(tongpoem_files)} file(s)")
        print(f"{'='*56}")
        for fp in tqdm(tongpoem_files, desc="Evaluating Tang poems"):
            res = _process_tongpoem_file(fp, tongpoem_evaluator, tongpoem_output_dir)
            if res:
                tongpoem_summary.append(res)
    else:
        print(f"[SKIP] Tongpoem: input dir missing")

    # --- Overall summary ---
    all_summary = {
        "songci": songci_summary,
        "tongpoem": tongpoem_summary,
    }
    summary_path = os.path.join(OUTPUT_ROOT, 'evaluation_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*56}")
    print(f"  Evaluation complete.")
    print(f"  Songci:  {len(songci_summary)} file(s) processed")
    print(f"  Tongpoem: {len(tongpoem_summary)} file(s) processed")
    print(f"  Summary saved to '{summary_path}'")
    print(f"{'='*56}")
