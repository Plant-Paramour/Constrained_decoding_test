import os
import json
from tqdm import tqdm
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

# --- 1. Load External Cipai Data ---
try:
    with open('data/cipai_data.json', 'r', encoding='utf-8') as f:
        CIPAI_DATA = json.load(f)
    ONE_SHOT_EXAMPLES = CIPAI_DATA['one_shot_examples']
    COMPLETION_DATA = CIPAI_DATA['completion_data']
    INSTRUCTION_DATA = CIPAI_DATA['instruction_data']
    print("Successfully loaded external Cipai data from 'cipai_data.json'.")
except FileNotFoundError:
    print("FATAL: 'cipai_data.json' not found.")
    exit()
except (KeyError, json.JSONDecodeError) as e:
    print(f"FATAL: 'cipai_data.json' is corrupted or missing required keys. Details: {e}")
    exit()

# --- 2. Configuration ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-4o"

THEMES = ["爱情与离愁", "悲伤与祭奠", "爱国与豪情", "山水田园", "哲理", "怀古"]
CIPAI = list(ONE_SHOT_EXAMPLES.keys())
PROMPT_TYPES = ["zero-shot", "one-shot", "completion", "instruction", "chain-of-thought"]

output_dir = "out"
os.makedirs(output_dir, exist_ok=True)
print(f"All outputs will be saved to: '{output_dir}'")

# --- 3. Completion Function ---
def generate_completion(messages, temperature=0.7, max_tokens=512):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None

# --- 4. Main Loop ---
for prompt_type in PROMPT_TYPES:
    print(f"\n=== Prompt Type: '{prompt_type}' ===")
    prompt_specific_results = []

    for theme in tqdm(THEMES, desc="Themes"):
        for cipai in CIPAI:
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
                else:
                    continue

            except KeyError:
                print(f"Skipping: Cipai '{cipai}' missing data for prompt type '{prompt_type}'.")
                continue

            result = generate_completion(messages)
            if result:
                prompt_specific_results.append({
                    "theme": theme,
                    "cipai": cipai,
                    "output": result
                })

    # Save results for this prompt type
    output_file = os.path.join(output_dir, f"gpt4o_{prompt_type}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(prompt_specific_results, f, ensure_ascii=False, indent=4)
    print(f"Saved: {output_file}")

print("\n All GPT-4o generations completed successfully.")
