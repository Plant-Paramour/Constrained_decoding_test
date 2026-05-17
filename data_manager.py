import json
import os
from typing import Dict, List, Tuple

class DataManager:
    def __init__(self, rhyme_dict_path: str, poem_path: str):
        self.rhyme_dict_path = rhyme_dict_path
        self.poem_path = poem_path

        self.char_to_rhyme_tone: Dict[str, List[Tuple[str, str]]] = {}
        self.rhyme_tone_to_chars: Dict[str, Dict[str, List[str]]] = {}

        # cipai_data: {词牌名: {"name": ..., "default_variant": ..., "variants": [...]}}
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
        if os.path.isdir(self.poem_path):
            # 新格式：每词牌一个 JSON 文件的目录
            for filename in os.listdir(self.poem_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.poem_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    name = data.get('name') or filename.replace('.json', '')
                    self.cipai_data[name] = data
        else:
            # 旧格式兼容：单一大 JSON 文件，自动转换为新格式
            with open(self.poem_path, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
            for name, entry in old_data.items():
                variant_name = entry.pop('variant', name)
                self.cipai_data[name] = {
                    "name": name,
                    "default_variant": variant_name,
                    "variants": [{
                        "name": variant_name,
                        **entry
                    }]
                }

    def get_cipai(self, name: str) -> Dict:
        """返回词牌的默认变体 dict（向后兼容旧调用方）"""
        if name not in self.cipai_data:
            raise ValueError(f"词牌 {name} 未找到。")
        data = self.cipai_data[name]
        variants = data.get('variants', [])
        if not variants:
            raise ValueError(f"词牌 {name} 没有变体数据。")
        default_name = data.get('default_variant', variants[0]['name'])
        for v in variants:
            if v['name'] == default_name:
                return v
        return variants[0]

    def get_cipai_variants(self, name: str) -> List[Dict]:
        """返回词牌的所有变体列表"""
        if name not in self.cipai_data:
            raise ValueError(f"词牌 {name} 未找到。")
        return self.cipai_data[name].get('variants', [])

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
