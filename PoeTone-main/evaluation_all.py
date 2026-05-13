import os
import json
import re
from collections import Counter
from pypinyin import pinyin, Style
from tqdm import tqdm

class SongciCritic:
    """
    A critic to evaluate generated Songci poems based on structural,
    tonal, and rhyme constraints from a metadata file.
    
    This version uses a "decoupled partial credit" system for a more
    nuanced and fair evaluation of model outputs.
    """
    def __init__(self, metadata_path):
        """
        Initializes the critic by loading Cipai metadata.
        
        Args:
            metadata_path (str): Path to the JSON metadata file.
        """
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.cipai_variants = {}
        for entry in self.metadata:
            cipai_name = entry['Cipai']
            if cipai_name not in self.cipai_variants:
                self.cipai_variants[cipai_name] = []
            self.cipai_variants[cipai_name].append(entry)

    def _parse_poem_into_lines(self, text: str) -> list[str]:
        """
        Robustly parses raw model output into a clean list of lines.
        Splits on common delimiters including commas and periods, and removes empty lines.
        """
        # Split on newline and common Chinese/English punctuation marks
        lines = re.split(r'[\n。！？；，,\.]', text)
        # Remove leading/trailing whitespace and filter out any resulting empty strings
        return [line.strip() for line in lines if line.strip()]

    def get_char_tone(self, char: str) -> str:
        """
        Gets the modern Mandarin tone for a character.
        Tones 1, 2 are '平' (Ping). Tones 3, 4 are '仄' (Ze).
        """
        try:
            tone = pinyin(char, style=Style.TONE3, heteronym=False)[0][0]
            last_char = tone[-1]
            if last_char in '12':
                return '平'
            elif last_char in '34':
                return '仄'
            return '中' # Neutral tone
        except (IndexError, TypeError):
            return '中' # Return neutral for non-mappable characters

    def get_char_rhyme(self, char: str) -> str:
        """
        Gets the modern Mandarin rhyme for a character as a proxy for rhyme group.
        """
        try:
            # Using the final part of the pinyin as a proxy
            p = pinyin(char, style=Style.FINALS, heteronym=False)[0][0]
            return p if p else 'none'
        except (IndexError, TypeError):
            return 'none'
            
    def _calculate_variant_score(self, generated_lines: list[str], variant_meta: dict) -> dict:
        """
        Calculates structure, tonal, and rhyme scores with a fair partial credit system.
        - Structure score measures completeness and correctness.
        - Tonal/Rhyme scores measure the quality of the generated part.
        """
        scores = {"structure": 0.0, "tonal": 0.0, "rhyme": 0.0}
        
        template_lines_chars = variant_meta.get('total', {}).get('chars_per_line', [])
        if not template_lines_chars:
            return scores

        n_template = len(template_lines_chars)
        n_generated = len(generated_lines)
        n_common = min(n_template, n_generated)

        # --- 1. Structure Score (Decoupled Partial Credit) ---
        if n_template > 0 or n_generated > 0:
            correct_structured_lines = sum(1 for i in range(n_common) if len(generated_lines[i]) == template_lines_chars[i])
            # Denominator is the larger of the two lengths to penalize both missing and extra lines
            denominator = max(n_template, n_generated, 1) # Use max(..., 1) to avoid division by zero
            scores['structure'] = correct_structured_lines / denominator

        # --- 2. Tonal Score (Decoupled Partial Credit) ---
        s1_lines = variant_meta.get('stanza1', {}).get('lines', [])
        s2_lines = variant_meta.get('stanza2', {}).get('lines', [])
        template_tonal_patterns = s1_lines + s2_lines if s1_lines else []
        
        if template_tonal_patterns:
            total_common_chars, matching_chars = 0, 0
            
            for i in range(n_common):
                if i >= len(template_tonal_patterns): break
                template_pattern = template_tonal_patterns[i].replace("句", "").replace("韵", "")
                if len(generated_lines[i]) != len(template_pattern): continue
                
                total_common_chars += len(generated_lines[i])
                for j, char in enumerate(generated_lines[i]):
                    required_tone = template_pattern[j]
                    actual_tone = self.get_char_tone(char)
                    if required_tone == '中' or actual_tone == required_tone:
                        matching_chars += 1
            
            # Denominator is the number of characters we could actually compare
            if total_common_chars > 0:
                scores['tonal'] = matching_chars / total_common_chars

        # --- 3. Rhyme Score (Decoupled Partial Credit) ---
        rhyme_positions = variant_meta.get('total', {}).get('rhyme_positions', [])
        if rhyme_positions:
            rhyming_chars = [generated_lines[pos - 1][-1] for pos in rhyme_positions if pos - 1 < n_generated and generated_lines[pos - 1]]
            
            if rhyming_chars:
                rhyme_groups = [self.get_char_rhyme(c) for c in rhyming_chars]
                valid_rhymes = [r for r in rhyme_groups if r != 'none']
                if valid_rhymes:
                    most_common_rhyme_count = Counter(valid_rhymes).most_common(1)[0][1]
                    # Denominator is the number of rhyming lines we could actually check
                    scores['rhyme'] = most_common_rhyme_count / len(rhyming_chars)
                    
        return scores

    def evaluate(self, cipai: str, output: str) -> dict:
        generated_lines = self._parse_poem_into_lines(output)
        variants = self.cipai_variants.get(cipai)
        if not variants:
            return {"error": f"Cipai '{cipai}' not found in metadata.", "total_score_percentage": 0.0}

        best_score = -1.0
        best_details = {"total_score_percentage": 0.0, "reason": "No valid variant found."}
        weights = {"S": 0.4, "T": 0.3, "R": 0.3}

        for i, variant in enumerate(variants):
            component_scores = self._calculate_variant_score(generated_lines, variant)
            total_score = (weights["S"] * component_scores["structure"] +
                           weights["T"] * component_scores["tonal"] +
                           weights["R"] * component_scores["rhyme"])
            
            if total_score > best_score:
                best_score = total_score
                best_details = {
                    "cipai": cipai,
                    "best_variant_index": i,
                    "total_score_percentage": round(total_score * 100, 2),
                    "structure_score_percentage": round(component_scores["structure"] * 100, 2),
                    "tonal_score_percentage": round(component_scores["tonal"] * 100, 2),
                    "rhyme_score_percentage": round(component_scores["rhyme"] * 100, 2),
                    "parsed_lines": generated_lines
                }
        return best_details


# --- Main execution block ---
if __name__ == "__main__":
    # --- Configuration ---
    METADATA_FILE = 'data/cipai_metadata.json'
    INPUT_DIR = 'out'
    OUTPUT_DIR = 'out'

    # --- Setup ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    critic = SongciCritic(METADATA_FILE)
    
    # --- List to hold the summary from all files ---
    overall_summary = []
    
    # --- Main Loop ---
    files_to_process = [f for f in os.listdir(INPUT_DIR) if f.endswith("_cleaned.json")]
    # files_to_process = [f for f in os.listdir(INPUT_DIR)]

    for filename in tqdm(files_to_process, desc="Processing all files"):
        input_filepath = os.path.join(INPUT_DIR, filename)
        
        with open(input_filepath, 'r', encoding='utf-8') as f:
            generated_poems = json.load(f)

        if not generated_poems:
            print(f"Skipping empty file: {filename}")
            continue

        individual_results = []
        total_score_sum, structure_score_sum, tonal_score_sum, rhyme_score_sum = 0, 0, 0, 0
        poem_count = 0

        for poem in generated_poems:
            evaluation = critic.evaluate(poem['cipai'], poem['output'])
            result_entry = {"original_poem": poem, "evaluation": evaluation}
            individual_results.append(result_entry)
            
            if "error" not in evaluation:
                total_score_sum += evaluation.get('total_score_percentage', 0)
                structure_score_sum += evaluation.get('structure_score_percentage', 0)
                tonal_score_sum += evaluation.get('tonal_score_percentage', 0)
                rhyme_score_sum += evaluation.get('rhyme_score_percentage', 0)
                poem_count += 1
        
        # Calculate Averages for the current file
        average_scores = {}
        if poem_count > 0:
            average_scores = {
                "average_total_score": round(total_score_sum / poem_count, 2),
                "average_structure_score": round(structure_score_sum / poem_count, 2),
                "average_tonal_score": round(tonal_score_sum / poem_count, 2),
                "average_rhyme_score": round(rhyme_score_sum / poem_count, 2),
            }
        else:
            average_scores = { "average_total_score": 0, "average_structure_score": 0, "average_tonal_score": 0, "average_rhyme_score": 0 }
            
        # Append this file's summary to the overall list
        file_summary = {
            "source_file": filename,
            "average_scores": average_scores
        }
        overall_summary.append(file_summary)
            
        # Prepare final JSON output for this specific file
        final_output_data = {
            "average_scores": average_scores,
            "individual_results": individual_results
        }
        
        # Save the detailed results for this file
        output_filename = filename.replace('_cleaned.json', '_evaluation_results.json')
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_output_data, f, ensure_ascii=False, indent=2)

    # Save the overall summary file
    overall_results_path = os.path.join(OUTPUT_DIR, 'overall_results.json')
    with open(overall_results_path, 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    print(f"\n Batch evaluation complete.")
    print(f"Individual results saved in '{OUTPUT_DIR}'")
    print(f"Overall summary saved to '{overall_results_path}'")