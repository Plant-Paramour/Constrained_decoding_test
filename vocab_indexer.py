import re
from typing import Dict, Set, Tuple, List
from transformers import PreTrainedTokenizer
from data_manager import DataManager
from tqdm import tqdm

class VocabIndexer:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_manager: DataManager):
        self.tokenizer = tokenizer
        self.data_manager = data_manager
        
        self.token_to_text: Dict[int, str] = {}
        # (length, pingze_pattern) -> set of token_ids
        # e.g., (2, "平仄") -> {id1, id2}
        self.pattern_tokens: Dict[Tuple[int, str], Set[int]] = {}
        
        # (rhyme_type(平/仄), rhyme_part) -> set of token_ids 
        self.rhyme_tokens: Dict[Tuple[str, str], Set[int]] = {}
        
        # Allow Chinese characters and English letters
        self.ch_pattern = re.compile(r'^[\u4e00-\u9fa5A-Za-z]+$')
        
        self._build_index()

    def _is_valid_text(self, text: str) -> bool:
        return bool(self.ch_pattern.match(text))

    def _get_all_pingze_patterns(self, text: str) -> List[str]:
        """返回多字词组合出的所有合法平仄序列。如有一个字不在韵书中，返回空。"""
        patterns = [""]
        for char in text:
            pz_list = self.data_manager.get_pingze(char)
            if not pz_list:
                return []
            new_patterns = []
            for pz in pz_list:
                for base in patterns:
                    new_patterns.append(base + pz)
            patterns = new_patterns
        return set(patterns)
        
    def _build_index(self):
        print("Building offline vocab index (Constraint rules)...")
        vocab = self.tokenizer.get_vocab()
        
        for token_str, token_id in tqdm(vocab.items()):
            # Qwen tokenizer (BPE) 返回的是字节编码或特殊格式，需解码获取真实字符
            clean_text = self.tokenizer.decode([token_id]).replace(' ', '')
            
            if not clean_text or not self._is_valid_text(clean_text):
                continue
                
            length = len(clean_text)
            self.token_to_text[token_id] = clean_text
            
            # 建平仄索引
            pz_patterns = self._get_all_pingze_patterns(clean_text)
            for pz in pz_patterns:
                key = (length, pz)
                if key not in self.pattern_tokens:
                    self.pattern_tokens[key] = set()
                self.pattern_tokens[key].add(token_id)
                
            # 建押韵索引 (只看最后一个字)
            last_char = clean_text[-1]
            if last_char in self.data_manager.char_to_rhyme_tone:
                for rhyme_part, pz in self.data_manager.char_to_rhyme_tone[last_char]:
                    r_key = (pz, rhyme_part)
                    if r_key not in self.rhyme_tokens:
                        self.rhyme_tokens[r_key] = set()
                    self.rhyme_tokens[r_key].add(token_id)
                    
        print(f"Indexed {len(self.token_to_text)} purely Chinese constraint tokens.")


