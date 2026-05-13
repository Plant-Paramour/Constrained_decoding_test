import os
import re
import json
from tqdm import tqdm

# --- Configuration ---
# Read from the original data location
INPUT_DIR = "data"
# Save to the new location to keep original files safe
OUTPUT_DIR = "out"

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define a set of keywords that indicate a line is part of a "chatty" preamble
CHATTY_KEYWORDS = {
    "我", "你", "分析", "创作", "用户", "希望", "首先", "接下来", "比如", "注意",
    "扮演", "角色", "要求", "主题", "格律", "声韵", "结构", "特点", "构思", "检查"
}

def clean_output(raw_text: str, cipai: str = None, theme: str = None) -> str:
    """
    Applies a robust, multi-stage cleaning process to raw model output.
    The core logic finds the last contiguous block of text that looks like a poem.
    """
    if not raw_text:
        return ""

    text = raw_text

    # --- Stage 1: Initial Block Removal (<think> tags) ---
    # This removes large, obvious chunks of non-poem text first.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*?</\think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # --- Stage 2: Isolate the Poem by Working Backwards ---
    # This is the core logic for removing conversational preambles.
    lines = text.strip().split('\n')
    poem_lines = []
    # Iterate from the end of the file upwards
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        
        # Check if the line contains conversational keywords.
        is_chatty = any(keyword in line for keyword in CHATTY_KEYWORDS)
        
        if is_chatty:
            # If we find a chatty line, we've reached the end of the poem. Stop.
            break
        else:
            # If it's not chatty, prepend it to our list of poem lines.
            poem_lines.insert(0, line)

    if not poem_lines:
        return "" # Return empty if no poem-like lines were found

    # --- Stage 3: Fine-grained Line Filtering ---
    # Now that we have the poem block, we apply final line-level cleaning.
    final_lines = []
    for line in poem_lines:
        # Per your rule, remove any line containing these characters.
        if '《' in line or '》' in line or '·' in line:
            continue
        # Also remove markers that are sometimes left over
        if "词作：" in line or "词作" in line:
            # If the marker is on its own line, skip. If not, clean it.
            clean_line = line.replace("词作：", "").replace("词作", "").strip()
            if clean_line:
                final_lines.append(clean_line)
            continue
            
        final_lines.append(line)

    final_text = "\n".join(final_lines)

    # --- Stage 4: Remove Footers ---
    final_text = re.split(r'\(注\)|（注）|注意', final_text, 1)[0]
    
    return final_text.strip()

def process_file(file_path, output_path):
    """Reads a raw JSON file, cleans each entry, and saves to a new file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading or decoding {file_path}: {e}")
        return

    cleaned_data = []
    for item in data:
        cipai = item.get("cipai", "")
        theme = item.get("theme", "")
        raw_output = item.get("output", "")
        
        cleaned_output = clean_output(raw_output, cipai=cipai, theme=theme)
        
        if cleaned_output:
            cleaned_data.append({
                "cipai": cipai,
                "theme": theme,
                "output": cleaned_output
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

def clean_all_json_files(input_dir, output_dir):
    """Finds all raw .json files and processes them."""
    files_to_process = [
        f for f in os.listdir(input_dir) 
        if f.endswith(".json") and not "_cleaned" in f and not "_evaluation_results" in f
    ]
    
    if not files_to_process:
        print("No raw .json files found to clean.")
        return

    for filename in tqdm(files_to_process, desc="Cleaning raw files"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.replace(".json", "_cleaned.json"))
        process_file(input_file, output_file)

if __name__ == "__main__":
    clean_all_json_files(INPUT_DIR, OUTPUT_DIR)
    print(f"\n Cleaning complete.")
    print(f"Original files in '{INPUT_DIR}' are untouched.")
    print(f"Cleaned files have been saved to '{OUTPUT_DIR}'.")