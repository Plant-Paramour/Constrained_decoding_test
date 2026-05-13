import json
import os
import gc

print("--- Script Starting: Initializing Environment ---")

BASE_DIR = "out"

os.environ["TRANSFORMERS_CACHE"] = os.path.join(BASE_DIR, "transformers_cache")
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_home")
os.environ["UNSLOTH_HOME"] = os.path.join(BASE_DIR, "unsloth_cache")
# Create the directories now to ensure they exist
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["UNSLOTH_HOME"], exist_ok=True)

from unsloth import FastLanguageModel
import transformers
import torch

# --- 1. Load External Cipai Data ---
print("Loading external Cipai data...")
try:
    with open('data/cipai_data.json', 'r', encoding='utf-8') as f:
        CIPAI_DATA = json.load(f)
    ONE_SHOT_EXAMPLES = CIPAI_DATA['one_shot_examples']
    COMPLETION_DATA = CIPAI_DATA['completion_data']
    INSTRUCTION_DATA = CIPAI_DATA['instruction_data']
    print("Successfully loaded external Cipai data from 'cipai_data.json'.")
except FileNotFoundError:
    print("FATAL ERROR: 'cipai_data.json' not found. Please create it first.")
    exit()
except (KeyError, json.JSONDecodeError) as e:
    print(f"FATAL ERROR: 'cipai_data.json' is corrupted or missing required keys. Details: {e}")
    exit()

# --- 2. Define Models and Generation Tasks ---
MODEL_IDS = [
    "unsloth/Qwen3-8B",
    "unsloth/Qwen3-32B",
    "unsloth/Qwen3-235B-A22B",
    "unsloth/Qwen3-4B",
    "unsloth/Qwen3-1.7B",
    "unsloth/Qwen3-0.6B",
]
# Define a list of huge models that require special handling.
HUGE_MODELS = ["unsloth/Qwen3-235B-A22B"]

THEMES = ["爱情与离愁", "悲伤与祭奠", "爱国与豪情", "山水田园", "哲理", "怀古"]
CIPAI = list(ONE_SHOT_EXAMPLES.keys())
PROMPT_TYPES = ["zero-shot", "one-shot", "completion", "instruction", "chain-of-thought"]

print("\n--- Experiment Setup ---")
print(f"Models to be tested: {len(MODEL_IDS)}")

output_dir = os.path.join(BASE_DIR, "generated_texts")
os.makedirs(output_dir, exist_ok=True)
print(f"All outputs will be saved to: '{output_dir}'")
print(f"All caches will be saved within: '{BASE_DIR}'")

print(f"Starting advanced prompt testing for {len(MODEL_IDS)} models.")
print(f"Testing on {len(CIPAI)} Cipai and {len(PROMPT_TYPES)} prompt types.")
print(f"A separate JSON file will be created for EACH model and EACH prompt type.")

# --- 3. Main Generation Loop ---

for model_id in MODEL_IDS:
    print("\n" + "="*80)
    print(f"Processing model: {model_id}")
    print("="*80)

    try:
        print(f"Attempting to load '{model_id}' via Unsloth...")

        # MODIFICATION: Use a dictionary to build loading arguments conditionally.
        loading_args = {
            "max_seq_length": 4096,
            "dtype": None,
            "load_in_4bit": True,
        }

        # Conditionally add the offloading flag ONLY for models in the HUGE_MODELS list.
        if model_id in HUGE_MODELS:
            print(f"INFO: '{model_id}' is a huge model. Enabling CPU offloading.")
            loading_args["llm_int8_enable_fp32_cpu_offload"] = True

        # Load the model by unpacking the dynamically created arguments dictionary.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            **loading_args
        )
        print(f"✅ Model and tokenizer for {model_id} loaded successfully.")

        pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
        print("✅ Transformers pipeline created successfully.")

        if pipeline.tokenizer.eos_token is None:
            stop_token = "<|im_end|>"
            if stop_token in pipeline.tokenizer.vocab:
                 print(f"⚠️ WARNING: Tokenizer's eos_token not set. Manually setting to '{stop_token}'.")
                 pipeline.tokenizer.eos_token = stop_token
            else:
                 fallback_token = "<|end_of_text|>"
                 if fallback_token in pipeline.tokenizer.vocab:
                    print(f"⚠️ WARNING: Tokenizer's eos_token not set. Manually setting to '{fallback_token}'.")
                    pipeline.tokenizer.eos_token = fallback_token

    except Exception as e:
        print(f"❌ FATAL: Error loading model {model_id}: {e}. Skipping to next model.")
        continue

    # --- Loop Through All Prompt Types and Tasks ---
    for prompt_type in PROMPT_TYPES:
        print(f"\n-- Testing Prompt Type: '{prompt_type}' --")
        
        prompt_specific_results = []
        for theme in THEMES:
            for cipai in CIPAI:
                messages = []
                try:
                    if prompt_type == "zero-shot":
                        messages = [
                            {"role": "system", "content": "你是一位宋代词人。请按照用户提供的词牌和主题创作一首词。请直接输出词作本身，不要包含任何标题、解释、或额外的对话文字。"},
                            {"role": "user", "content": f"请以《{cipai}》为词牌，以“{theme}”为主题，创作一首宋词。"}
                        ]
                    elif prompt_type == "one-shot":
                        example = ONE_SHOT_EXAMPLES[cipai]
                        messages = [
                            {"role": "system", "content": "你是一位宋代词人，擅长模仿范例进行创作。请按照用户提供的词牌、主题以及范例创作一首词。请直接输出词作本身，不要包含任何标题、解释、或额外的对话文字。"},
                            {"role": "user", "content": f"这是一首以《{cipai}》为词牌的范例：\n\n{example}\n\n现在，请模仿这首词的风格和格律，以“{theme}”为主题，创作一首全新的词。"}
                        ]
                    elif prompt_type == "completion":
                        first_half = COMPLETION_DATA[cipai]['first_half']
                        messages = [
                            {"role": "system", "content": "你是一位宋代词人，擅长续写词作。请按照用户提供的上阕创作补齐下阕，下阕须符合用户提供的词牌和主题。请直接输出词作本身，不要包含任何标题、解释、或额外的对话文字。"},
                            {"role": "user", "content": f"这是著名词牌《{cipai}》的上阕：\n\n{first_half}\n\n请你以此为开篇，围绕“{theme}”这一主题，创作一个全新的下阕。内容完全原创，不得与原词下阕雷同。"}
                        ]
                    elif prompt_type == "instruction":
                        rules = INSTRUCTION_DATA[cipai]
                        messages = [
                            {"role": "system", "content": "你是一位宋代词人。请根据用户提供的词牌、主题以及格律要求创作一首词。请直接输出词作本身，不要包含任何标题、解释、或额外的对话文字。"},
                            {"role": "user", "content": f"请为我创作一首词。主题为“{theme}”，词牌为《{cipai}》。\n你必须严格遵守以下格律要求：\n{rules}\n请直接开始创作。"}
                        ]
                    elif prompt_type == "chain-of-thought":
                        messages = [
                            {"role": "system", "content": "你是一位宋代词人。请根据用户提供的词牌和主题创作一首词。请先用一句话描述用户提供词牌的格律、声韵以及结构特点。之后请输出词作本身，不要包含任何标题、解释、或额外的对话文字。"},
                            {"role": "user", "content": f"请为我创作一首词牌为《{cipai}》，主题为“{theme}”的宋词。在创作之前，请先用几句话分析并阐述《{cipai}》的格律、声韵和结构特点。然后，另起一段，再呈现你的完整词作。\n请按“分析”和“词作”两个部分清晰地组织你的回答。"}
                        ]
                except KeyError:
                    print(f"  > SKIPPING: No data available for Cipai '{cipai}' in prompt type '{prompt_type}'."); continue

                print(f"  > Generating for Theme: '{theme}', Cipai: '{cipai}'...")

                prompt = pipeline.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                )
                
                terminators = []
                if pipeline.tokenizer.eos_token_id is not None:
                    terminators.append(pipeline.tokenizer.eos_token_id)
                
                im_end_token = "<|im_end|>"
                if im_end_token in pipeline.tokenizer.vocab:
                    terminators.append(pipeline.tokenizer.convert_tokens_to_ids(im_end_token))
                
                terminators = list(set(terminators)) if terminators else None

                outputs = pipeline(
                    prompt, max_new_tokens=1024, eos_token_id=terminators,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=pipeline.tokenizer.eos_token_id
                )
                generated_text = outputs[0]["generated_text"][len(prompt):].strip()
                prompt_specific_results.append({
                    "theme": theme, "cipai": cipai, "output": generated_text
                })

        # --- SAVE results for the current model and prompt type ---
        if prompt_specific_results:
            sanitized_model_id = model_id.split('/')[-1]
            output_filename = f"{sanitized_model_id}_{prompt_type}.json"

            output_filepath = os.path.join(output_dir, output_filename)

            print(f"  > Saving results to '{output_filepath}'...")
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(prompt_specific_results, f, ensure_ascii=False, indent=4)
            print(f"  > Save complete.")

    print(f"\n--- Releasing Memory for {model_id} ---")
    del pipeline, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ Memory released.")

print("\n" + "="*80)
print("✅ All models and advanced prompts processed successfully.")
print("Check your directory for the generated JSON files.")