import json
import re
import os
from collections import Counter
from pypinyin import pinyin, Style
from tqdm import tqdm


class SongciEvaluator:
    """
    Evaluate generated Songci based on structure, tonal, and rhyme constraints.
    Adapted for test/Meter/songci.json format.
    """

    def __init__(self, meter_path):
        with open(meter_path, 'r', encoding='utf-8') as f:
            self.meters = json.load(f)

    def _parse_poem_into_lines(self, text):
        lines = re.split(r'[。，；！？、\n]', text)
        return [line.strip() for line in lines if line.strip()]

    @staticmethod
    def get_char_tone(char):
        try:
            tone_char = pinyin(char, style=Style.TONE3, heteronym=False)[0][0][-1]
            if tone_char in '12':
                return '平'
            elif tone_char in '34':
                return '仄'
            return '中'
        except (IndexError, TypeError):
            return '中'

    @staticmethod
    def get_char_rhyme(char):
        try:
            fin = pinyin(char, style=Style.FINALS, heteronym=False)[0][0]
            return fin if fin else 'none'
        except (IndexError, TypeError):
            return 'none'

    def _build_total_view(self, meter_entry):
        """Build a consolidated view, splitting template lines by 、 for clause-level matching."""
        chars_per_line = []
        global_rhyme_positions = []
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

            for key in stanza:
                if key.startswith('rhyme_') and key.endswith('_positions'):
                    for pos in stanza[key]:
                        line_idx = pos - 1
                        if line_idx < len(line_to_subclauses):
                            sub_indices = line_to_subclauses[line_idx]
                            if sub_indices:
                                global_pos = sub_indices[-1] + 1
                                if global_pos not in global_rhyme_positions:
                                    global_rhyme_positions.append(global_pos)

        global_rhyme_positions.sort()
        return {
            'chars_per_line': chars_per_line,
            'rhyme_positions': global_rhyme_positions,
            'tonal_patterns': tonal_patterns,
        }

    def _get_template_line_count(self, meter_entry):
        """Get total number of sub-clauses (after 、 splitting) across all stanzas."""
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
        rhyme_positions = total['rhyme_positions']

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
                    actual = self.get_char_tone(char)
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
        if rhyme_positions:
            rhyming_chars = []
            rhyme_group_map = {}
            for pos in rhyme_positions:
                idx = pos - 1
                if idx < n_generated and generated_lines[idx]:
                    char = generated_lines[idx][-1]
                    fin = self.get_char_rhyme(char)
                    rhyming_chars.append(char)
                    if fin != 'none':
                        rhyme_group_map.setdefault(fin, []).append({
                            "position": pos,
                            "char": char,
                        })
            if rhyming_chars:
                rhyme_finals = [self.get_char_rhyme(c) for c in rhyming_chars]
                valid = [r for r in rhyme_finals if r != 'none']
                if valid:
                    most_common = Counter(valid).most_common(1)[0]
                    scores['rhyme'] = most_common[1] / len(rhyming_chars)
                    details['rhyme'] = {
                        "rhyme_groups": rhyme_group_map,
                        "dominant_final": most_common[0],
                        "dominant_count": most_common[1],
                        "total_rhyme_positions": len(rhyming_chars),
                    }
                else:
                    details['rhyme'] = {
                        "rhyme_groups": {},
                        "dominant_final": None,
                        "dominant_count": 0,
                        "total_rhyme_positions": len(rhyming_chars),
                    }

        return scores, details

    def evaluate(self, cipai, output):
        generated_lines = self._parse_poem_into_lines(output)
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
#  Utility: extract poem text from raw AI output
# ============================================================

def extract_poem_text(raw_output):
    """
    Strictly extract poem body from [content] marker.
    Returns the poem text or None if no [content] marker found.
    """
    m = re.search(r'\[content\]\s*(.+)', raw_output, re.DOTALL)
    if m:
        text = m.group(1).strip()
        return text if text else None
    return None


# ============================================================
#  Main — batch evaluation
# ============================================================

if __name__ == "__main__":
    import argparse

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Evaluate generated Songci against meter rules")
    parser.add_argument("--meter", default=os.path.join(SCRIPT_DIR, "Meter", "songci.json"),
                        help="Path to meter JSON file")
    parser.add_argument("--input", default=os.path.join(SCRIPT_DIR, "input"),
                        help="Input directory or JSON file containing generated poems")
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "output"),
                        help="Output directory for evaluation results")
    parser.add_argument("--cipai", default=None,
                        help="Cipai name override (extracted from filename if not given)")
    args = parser.parse_args()

    METER_FILE = args.meter
    INPUT_PATH = args.input
    OUTPUT_DIR = args.output

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    evaluator = SongciEvaluator(METER_FILE)
    print(f"Loaded {len(evaluator.meters)} cipai from '{METER_FILE}'")

    # Collect input files
    if os.path.isfile(INPUT_PATH):
        input_files = [INPUT_PATH]
        input_dir = os.path.dirname(INPUT_PATH) or '.'
    else:
        input_dir = INPUT_PATH
        input_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith('.txt')
        ]

    overall_summary = []

    for filepath in tqdm(input_files, desc="Evaluating poems"):
        filename = os.path.basename(filepath)

        # Infer cipai from filename (e.g. "浣溪沙.txt" → "浣溪沙")
        cipai = args.cipai
        if not cipai:
            cipai = re.sub(r'[\._\-·].*', '', filename)
            cipai = cipai.strip()

        if cipai not in evaluator.meters:
            print(f"  [SKIP] '{cipai}' (from '{filename}') not in meter data")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Split into individual works by === 作品 N === markers
        works = re.split(r'===+\s*作品\s*\d+\s*===+', raw_text)
        if len(works) <= 1:
            # No markers found, treat entire file as one poem
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
        out_path = os.path.join(OUTPUT_DIR, out_name)
        output_data = {
            "source_file": filename,
            "cipai": cipai,
            "average_scores": avg_scores,
            "individual_results": poem_results,
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        overall_summary.append({
            "source_file": filename,
            "cipai": cipai,
            "works_evaluated": poem_count,
            "average_scores": avg_scores,
        })

    # Save overall summary
    summary_path = os.path.join(OUTPUT_DIR, 'evaluation_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    print(f"\nEvaluation complete. {len(overall_summary)} files processed.")
    print(f"Summary saved to '{summary_path}'")
