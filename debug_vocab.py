import torch
from transformers import AutoTokenizer
from data_manager import DataManager
from vocab_indexer import VocabIndexer
import re

def test_vocab_indexer():
    model_name = r"C:\Users\26051\.cache\modelscope\hub\models\Qwen\Qwen3-4B"
    print(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print("\n--- 1. Testing Tokenizer Decoding ---")
    ch_pattern = re.compile(r'^[\u4e00-\u9fa5A-Za-z]+$')
    
    count_raw_match = 0
    count_decoded_match = 0
    
    # Check first 5000 tokens as a sample
    for token_id in range(5000):
        # some tokenizers use convert_ids_to_tokens, some use decode
        raw_token = tokenizer.convert_ids_to_tokens(token_id)
        if raw_token is None:
            continue
            
        decoded_token = tokenizer.decode([token_id])
        
        is_raw_match = ch_pattern.match(raw_token) is not None
        is_decoded_match = ch_pattern.match(decoded_token) is not None
        
        if is_raw_match:
            count_raw_match += 1
        if is_decoded_match:
            count_decoded_match += 1
            if count_decoded_match <= 10:
                print(f"Token ID {token_id}: Raw='{raw_token}' -> Decoded='{decoded_token}'")
                
    print(f"Sample 5000 tokens:")
    print(f"  Raw token string match count: {count_raw_match}")
    print(f"  Decoded token string match count: {count_decoded_match}")
    
    print("\n--- 2. Testing DataManager (Pingze & Rhyme) ---")
    try:
        dm = DataManager(rhyme_dict_path="Rhyme/Cilin.json", songci_path="Meter/songci.json")
        pz_chun = dm.get_pingze("春")
        pz_hua = dm.get_pingze("花")
        print(f"春的平仄: {pz_chun}")
        print(f"花的平仄: {pz_hua}")
    except Exception as e:
        print(f"DataManager Error: {e}")

if __name__ == "__main__":
    test_vocab_indexer()

