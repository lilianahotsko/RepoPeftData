import json, os
from collections import Counter

baselines = os.environ['SCRATCH'] + '/BASELINES'
icl = json.load(open(baselines + '/icl_3shot_cr_test.json'))
pre = json.load(open(baselines + '/pretrained_cr_test.json'))

pre_map = {(e['repo'], e['expected']): e for e in pre['entries']}

wrong_preds = Counter()
pattern_counts = Counter()
for e in icl['entries']:
    key = (e['repo'], e['expected'])
    pe = pre_map.get(key)
    if not pe or not pe.get('exact_match', False) or e.get('exact_match', False):
        continue
    got = e.get('got', '')
    wrong_preds[got] += 1
    if got == '':
        pattern_counts['empty'] += 1
    elif got in ('None', '0', '1', 'True', 'False'):
        pattern_counts['generic_value'] += 1
    elif 'assert' in got.lower():
        pattern_counts['contains_assert'] += 1
    else:
        pattern_counts['other_wrong'] += 1

print('ICL failure patterns:')
for pat, count in pattern_counts.most_common():
    print('  %s: %d' % (pat, count))
print('Total: %d' % sum(pattern_counts.values()))
print()
print('Top wrong predictions:')
for pred, count in wrong_preds.most_common(15):
    print('  %4d  %s' % (count, repr(pred)))
