import json
from typing import Dict, List, Tuple

class DataManager:
    def __init__(self, rhyme_dict_path: str, poem_path: str):
        self.rhyme_dict_path = rhyme_dict_path
        self.poem_path = poem_path
        
        self.char_to_rhyme_tone: Dict[str, List[Tuple[str, str]]] = {}
        self.rhyme_tone_to_chars: Dict[str, Dict[str, List[str]]] = {}
        
        self.cipai_data: Dict = {}
        
        self._load_rhyme()
        self._load_poem()

    def _tone_to_pingze(self, tone: str) -> str:
        if "平" in tone:
            return "平"
        return "仄"

    def _load_rhyme(self):
        with open(self.rhyme_dict_path, 'r', encoding='utf-8') as f:
            rhyme_data = json.load(f)
            
        for rhyme_part, tones in rhyme_data.items():
            self.rhyme_tone_to_chars[rhyme_part] = {}
            for tone_name, chars in tones.items():
                pingze = self._tone_to_pingze(tone_name)
                
                if pingze not in self.rhyme_tone_to_chars[rhyme_part]:
                    self.rhyme_tone_to_chars[rhyme_part][pingze] = []
                self.rhyme_tone_to_chars[rhyme_part][pingze].extend(chars)
                
                for char in chars:
                    if char not in self.char_to_rhyme_tone:
                        self.char_to_rhyme_tone[char] = []
                    self.char_to_rhyme_tone[char].append((rhyme_part, pingze))
                    
    def _load_poem(self):
        with open(self.poem_path, 'r', encoding='utf-8') as f:
            self.cipai_data = json.load(f)
            
    def get_cipai(self, name: str) -> Dict:
        if name not in self.cipai_data:
            raise ValueError(f"词牌 {name} 未找到。")
        return self.cipai_data[name]

    def get_pingze(self, char: str) -> List[str]:
        """获取一个字的所有可能的平仄"""
        if char not in self.char_to_rhyme_tone:
            return []
        return list(set([item[1] for item in self.char_to_rhyme_tone[char]]))

    def get_rhyme_part(self, char: str) -> List[str]:
        """获取一个字的所有可能的韵部"""
        if char not in self.char_to_rhyme_tone:
            return []
        return list(set([item[0] for item in self.char_to_rhyme_tone[char]]))

    def get_rhyme_part_by_tone(self, char: str, tone_pz: str) -> List[str]:
        """获取一个字的所有可能的且平仄匹配的韵部"""
        if char not in self.char_to_rhyme_tone:
            return []
        return list(set([item[0] for item in self.char_to_rhyme_tone[char] if item[1] == tone_pz]))
