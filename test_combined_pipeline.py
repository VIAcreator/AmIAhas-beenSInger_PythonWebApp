"""测试正则+Qwen 联合管道在全部标注数据上的准确率。"""
import csv, random
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report
from modules.content_classifier import classify
from modules.llm_verifier import QwenVerifier

random.seed(42)

# 1. 加载标注数据
samples = []
for fname, label in [('music.csv','ia_music'),('irrelevent.csv','irrelevant'),('related.csv','ia_related')]:
    with open(f'data/labeled_samples/{fname}') as f:
        reader = csv.reader(f)
        header = next(reader)
        ti, gi = header.index('title'), header.index('tags')
        ci = header.index('category') if 'category' in header else -1
        for r in reader:
            if len(r) > max(ti,gi):
                samples.append({
                    'title': r[ti], 'tags': r[gi],
                    'category': r[ci] if ci>=0 and ci<len(r) else '',
                    'content_type': label,
                })

print(f'标注总数: {len(samples)}')
for lb in ['ia_music','ia_related','irrelevant']:
    print(f'  {lb}: {sum(1 for s in samples if s["content_type"]==lb)}')
print()

# 2. 正则分类
regex_correct = 0
suspicious_samples = []
for s in samples:
    r = classify(s['title'], s['tags'], s['category'])
    s['regex_pred'] = r['content_type']
    s['suspicious'] = r['suspicious']
    s['is_game'] = r['is_game']
    s['is_cover'] = r['is_cover']
    if r['content_type'] == s['content_type']:
        regex_correct += 1
    if r['suspicious']:
        suspicious_samples.append(s)

print(f'[正则] 准确率: {regex_correct}/{len(samples)} ({regex_correct/len(samples):.1%})')
susp_correct = sum(1 for s in suspicious_samples if s['regex_pred'] == s['content_type'])
print(f'[正则] 可疑: {len(suspicious_samples)} 条, 其中正确 {susp_correct} ({susp_correct/len(suspicious_samples):.1%})')
print()

# 3. Qwen 验证可疑
print(f'[Qwen] 验证 {len(suspicious_samples)} 条可疑...')
verifier = QwenVerifier()
qwen_modified = 0
for i, s in enumerate(suspicious_samples):
    result = verifier.verify_one(s)
    s['final_pred'] = result['content_type']
    if result['content_type'] != s['regex_pred']:
        qwen_modified += 1
    if (i+1) % 50 == 0:
        print(f'  {i+1}/{len(suspicious_samples)}')

for s in samples:
    if not s['suspicious']:
        s['final_pred'] = s['regex_pred']

print(f'[Qwen] 修改: {qwen_modified}/{len(suspicious_samples)}')
print()

# 4. 最终结果
final_correct = sum(1 for s in samples if s['final_pred'] == s['content_type'])
print(f'[最终] 准确率: {final_correct}/{len(samples)} ({final_correct/len(samples):.1%})')

trues = [s['content_type'] for s in samples]
preds = [s['final_pred'] for s in samples]
print()
print(classification_report(trues, preds, labels=['ia_music','ia_related','irrelevant'], zero_division=0))

# 5. 分析
regex_errors = [s for s in samples if s['regex_pred'] != s['content_type']]
qwen_fixed = sum(1 for s in regex_errors if s['final_pred'] == s['content_type'])
qwen_broke = sum(1 for s in samples if s['regex_pred'] == s['content_type'] and s['final_pred'] != s['content_type'])
print(f'正则错误 {len(regex_errors)} 条 -> Qwen纠正 {qwen_fixed} 条, Qwen误改 {qwen_broke} 条')

print('\nQwen纠正示例:')
count = 0
for s in regex_errors:
    if s['final_pred'] == s['content_type']:
        print(f'  [{s["regex_pred"]} -> {s["final_pred"]}] {s["title"][:60]}')
        count += 1
        if count >= 8: break
