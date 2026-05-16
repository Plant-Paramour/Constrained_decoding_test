import json
from pypinyin import pinyin, Style
from collections import defaultdict

with open('C:/code/Constrained_decoding_test_bio/evaluation/Rhyme/Xinyun.json', 'r', encoding='utf-8') as f:
    xinyun = json.load(f)

XINYUN_FINAL_MAP = {
    'a': '一麻', 'ia': '一麻', 'ua': '一麻',
    'o': '二波', 'e': '二波', 'uo': '二波',
    'ie': '三皆', 'ue': '三皆', 've': '三皆',
    'ai': '四开', 'uai': '四开',
    'ei': '五微', 'ui': '五微', 'uei': '五微',
    'ao': '六豪', 'iao': '六豪',
    'ou': '七尤', 'iu': '七尤', 'iou': '七尤',
    'an': '八寒', 'ian': '八寒', 'uan': '八寒', 'van': '八寒',
    'en': '九文', 'in': '九文', 'un': '九文', 'uen': '九文', 'vn': '九文',
    'ang': '十唐', 'iang': '十唐', 'uang': '十唐',
    'eng': '十一庚', 'ing': '十一庚', 'ong': '十一庚', 'iong': '十一庚', 'ueng': '十一庚',
    'er': '十二齐',
    'v': '十二齐',
}
_ZHI_INITIALS = {'zh', 'ch', 'sh', 'r', 'z', 'c', 's'}
_JU_INITIALS = {'j', 'q', 'x', 'y'}

def get_char_rhyme(char):
    try:
        initial = pinyin(char, style=Style.INITIALS, heteronym=False)[0][0]
        final = pinyin(char, style=Style.FINALS, heteronym=False)[0][0]
        if not final:
            return 'none'
    except:
        return 'none'
    if final == 'i':
        if initial in _ZHI_INITIALS:
            return '十三支'
        return '十二齐'
    if final == 'u':
        if initial in _JU_INITIALS:
            return '十二齐'
        return '十四姑'
    return XINYUN_FINAL_MAP.get(final, final)

mismatches = defaultdict(list)
correct = 0
total = 0

for expected_category in xinyun:
    for tone in ['平', '仄']:
        for char in xinyun[expected_category].get(tone, []):
            total += 1
            result = get_char_rhyme(char)
            if result == expected_category:
                correct += 1
            else:
                mismatches[(expected_category, result)].append(char)

out = 'C:/code/Constrained_decoding_test_bio/evaluation/mismatch_report.txt'
with open(out, 'w', encoding='utf-8') as f:
    f.write(f'Total: {total}  Correct: {correct}  Mismatch: {total-correct}\n')
    f.write(f'Accuracy: {correct/total*100:.2f}%\n\n')
    for (exp, act), chars in sorted(mismatches.items()):
        f.write(f'{exp} => {act}: {len(chars)} chars\n')
        f.write(f'  {chars}\n\n')

print(f'Report saved to {out}')
print(f'Accuracy: {correct}/{total} = {correct/total*100:.2f}%')
