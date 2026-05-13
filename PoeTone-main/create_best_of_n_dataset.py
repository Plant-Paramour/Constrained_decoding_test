import os
import json
import re
from collections import Counter
from pypinyin import pinyin, Style
from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ======================================================================================
#  Configuration
# ======================================================================================
# --- How many candidates to generate for each prompt ---
N_CANDIDATES = 4

# --- File Paths ---
SCRATCH_WORKSPACE = "out"
CACHE_DIR = os.path.join(SCRATCH_WORKSPACE, 'huggingface_cache')
os.environ['HF_HOME'] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_FILE = os.path.join(SCRIPT_DIR, 'data/cipai_metadata.json')
PROMPT_FILE = os.path.join(SCRIPT_DIR, 'prompts.csv')

# --- Output Dataset File ---
# This is the high-quality dataset we are creating
OUTPUT_DATASET_FILE = os.path.join(SCRATCH_WORKSPACE, "best_of_n_dataset_Qwen.jsonl")

# --- Model Configuration ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" # Keeping your model name

# ======================================================================================
#  The SongciCritic Class
# ======================================================================================
class SongciCritic:
    # ... (The full SongciCritic class) ...
    def __init__(self, metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.cipai_variants = {}
        for entry in self.metadata:
            cipai_name = entry['Cipai']
            self.cipai_variants.setdefault(cipai_name, []).append(entry)
    def _parse_poem_into_lines(self, text: str) -> list[str]:
        lines = re.split(r'[\n。！？；，,\.]', text)
        return [line.strip() for line in lines if line.strip()]
    def get_char_tone(self, char: str) -> str:
        try:
            tone = pinyin(char, style=Style.TONE3, heteronym=False)[0][0][-1]
            return '平' if tone in '12' else '仄' if tone in '34' else '中'
        except: return '中'
    def get_char_rhyme(self, char: str) -> str:
        try:
            p = pinyin(char, style=Style.FINALS, heteronym=False)[0][0]
            return p if p else 'none'
        except: return 'none'
    def _calculate_variant_score(self, generated_lines: list[str], variant_meta: dict) -> dict:
        scores = {"structure": 0.0, "tonal": 0.0, "rhyme": 0.0}
        template_lines_chars = variant_meta.get('total', {}).get('chars_per_line', [])
        if not template_lines_chars: return scores
        n_template, n_generated = len(template_lines_chars), len(generated_lines)
        n_common = min(n_template, n_generated)
        if n_template > 0 or n_generated > 0:
            correct_lines = sum(1 for i in range(n_common) if len(generated_lines[i]) == template_lines_chars[i])
            scores['structure'] = correct_lines / max(n_template, n_generated, 1)
        template_tonal_patterns = variant_meta.get('stanza1', {}).get('lines', []) + variant_meta.get('stanza2', {}).get('lines', [])
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
                    if required_tone == '中' or actual_tone == required_tone: matching_chars += 1
            scores['tonal'] = matching_chars / total_common_chars if total_common_chars > 0 else 0.0
        rhyme_positions = variant_meta.get('total', {}).get('rhyme_positions', [])
        rhyming_chars = [generated_lines[pos - 1][-1] for pos in rhyme_positions if pos - 1 < n_generated and generated_lines[pos - 1]]
        if rhyming_chars:
            valid_rhymes = [r for r in [self.get_char_rhyme(c) for c in rhyming_chars] if r != 'none']
            if valid_rhymes:
                most_common_count = Counter(valid_rhymes).most_common(1)[0][1]
                scores['rhyme'] = most_common_count / len(rhyming_chars)
        return scores
    def evaluate(self, cipai: str, output: str) -> dict:
        generated_lines = self._parse_poem_into_lines(output)
        variants = self.cipai_variants.get(cipai)
        if not variants: return {"error": f"Cipai '{cipai}' not found."}
        best_score, best_details = -1.0, {"total_score": 0.0}
        weights = {"S": 0.4, "T": 0.3, "R": 0.3}
        for variant in variants:
            cs = self._calculate_variant_score(generated_lines, variant)
            total_score = (weights["S"] * cs["structure"] + weights["T"] * cs["tonal"] + weights["R"] * cs["rhyme"])
            if total_score > best_score:
                best_score = total_score
                best_details = {
                    "total_score": round(total_score, 4),
                    "structure_score": round(cs["structure"], 4),
                    "tonal_score": round(cs["tonal"], 4),
                    "rhyme_score": round(cs["rhyme"], 4),
                }
        return best_details

# ======================================================================================
#  Main Dataset Creation Logic
# ======================================================================================
if __name__ == "__main__":
    print("Starting 'Best-of-N' dataset creation process...")

    # --- Load Model for Generation ---
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, cache_dir=CACHE_DIR
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True)
    
    # --- Manually set the chat template for the Qwen model ---
    tokenizer.chat_template = (
        "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
                "{{'<|im_start|>system\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% elif message['role'] == 'user' %}"
                "{{'<|im_start|>user\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% elif message['role'] == 'assistant' %}"
                "{{'<|im_start|>assistant\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load Critic and Prompts ---
    critic = SongciCritic(METADATA_FILE)
    prompt_df = pd.read_csv(PROMPT_FILE, header=None, names=['cipai', 'theme'])
    prompts = prompt_df.to_dict('records')
    print(f"Loaded {len(prompts)} prompts to generate from.")

    generation_kwargs = {
        "top_k": 50, "top_p": 0.9, "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id, "max_new_tokens": 200,
        "num_return_sequences": N_CANDIDATES # Generate N candidates per prompt
    }

    # --- Open the output file and start the loop ---
    # with open(OUTPUT_DATASET_FILE, 'w', encoding='utf-8') as f:
    with open(OUTPUT_DATASET_FILE, 'a', encoding='utf-8') as f:
        # The cleaning function from our previous discussion
        # This is now integrated into the loop
        for prompt_data in tqdm(prompts, desc="Processing Prompts"):
            messages = [{"role": "user", "content": f"请以《{prompt_data['cipai']}》为词牌，以“{prompt_data['theme']}”为主题，创作一首宋词。请直接输出词作本身，不要包含任何标题、解释、或额外的对话文字。"}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokenized_prompt = tokenizer(prompt_text, return_tensors="pt").to(model.device)

            response_tensors = model.generate(tokenized_prompt["input_ids"], **generation_kwargs)
            
            full_texts = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            candidates = [text[len(prompt_text.replace(tokenizer.bos_token, '')):] for text in full_texts]

            best_candidate = None
            best_score = -1.0

            for cand in candidates:
                # The cleaning function is not needed here as we assume the critic handles parsing
                eval_result = critic.evaluate(prompt_data['cipai'], cand)
                score = eval_result.get('total_score', 0.0)
                if score > best_score:
                    best_score = score
                    best_candidate = cand
            
            if best_candidate:
                sft_record = {
                    "messages": [
                        {"role": "user", "content": messages[0]['content']},
                        {"role": "assistant", "content": best_candidate.strip()}
                    ]
                }
                f.write(json.dumps(sft_record, ensure_ascii=False) + "\n")

    print(f"\n Dataset creation complete. High-quality dataset saved to '{OUTPUT_DATASET_FILE}'")