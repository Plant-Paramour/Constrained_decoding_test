import json
import os

input_path = r'D:\[CODE]\Constrained_decoding_test\Rhyme\Cilin.json'
output_path = r'/PolyphonicChars/CilinPolyphonic.json'

with open(input_path, 'r', encoding='utf-8') as f:
    cilin_data = json.load(f)

# { '第一部': { '平声': ['东', '冬'], '仄声': ['董', '肿'] }, ... }
# Note: JSON structure might slightly vary, let's track everything.

char_records = {}

# Assuming structure is {category: {tone: [chars]}}
# We extract characters and track their categories and tones.
if isinstance(cilin_data, dict):
    for category, tones in cilin_data.items():
        if isinstance(tones, dict):
            for tone, chars in tones.items():
                if isinstance(chars, list):
                    for char in chars:
                        if char not in char_records:
                            char_records[char] = []
                        char_records[char].append({
                            'category': category,
                            'tone': tone
                        })

polyphonic_chars = {}
for char, records in char_records.items():
    # Deduplicate in case a char is listed multiple times in the exactly same category & tone
    unique_records = []
    for r in records:
        if r not in unique_records:
            unique_records.append(r)
    
    if len(unique_records) > 1:
        polyphonic_chars[char] = unique_records

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(polyphonic_chars, f, indent=4, ensure_ascii=False)

print(f"Extracted {len(polyphonic_chars)} polyphonic characters to {output_path}")

