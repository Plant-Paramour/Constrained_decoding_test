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

def map_final_to_xinyun(initial, final):
    if not final:
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

def get_all_xinyun_categories(char):
    try:
        all_initials = pinyin(char, style=Style.INITIALS, heteronym=True)[0]
        all_finals = pinyin(char, style=Style.FINALS, heteronym=True)[0]
    except:
        return set()

    n = min(len(all_initials), len(all_finals))
    categories = set()
    for i in range(n):
        cat = map_final_to_xinyun(all_initials[i], all_finals[i])
        if cat != 'none':
            categories.add(cat)
    return categories

mismatches = defaultdict(list)
correct = 0
total = 0

for expected_category in xinyun:
    for tone in ['平', '仄']:
        for char in xinyun[expected_category].get(tone, []):
            total += 1
            all_cats = get_all_xinyun_categories(char)
            if not all_cats:
                mismatches[('NO_READING', '')].append(char)
            elif expected_category in all_cats:
                correct += 1
            else:
                mismatches[(expected_category, tuple(sorted(all_cats)))].append(char)

out = 'C:/code/Constrained_decoding_test_bio/evaluation/mismatch_report_v3.txt'
with open(out, 'w', encoding='utf-8') as f:
    f.write(f'=== Heteronym=True validation (fixed zip) ===\n\n')
    f.write(f'Total: {total}  Correct: {correct}  Mismatch: {total-correct}\n')
    f.write(f'Accuracy: {correct/total*100:.2f}%\n\n')
    if mismatches:
        for (exp, all_cats), chars in sorted(mismatches.items()):
            f.write(f'{exp} => {all_cats}\n')
            f.write(f'  {len(chars)} chars: {chars}\n\n')

print(f'Accuracy: {correct}/{total} = {correct/total*100:.2f}%')
print(f'Report: {out}')
