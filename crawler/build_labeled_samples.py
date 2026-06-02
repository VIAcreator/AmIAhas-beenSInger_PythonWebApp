"""
将 labeled_samples/ 下的分类 CSV 合并为 QLoRA 训练用 JSON。

每个 CSV 文件名 = content_type 标签。
只保留 title 和 tags 列，其余列丢弃。
输出: data/labeled_samples.json
"""

import os
import json
import pandas as pd

SRC_DIR = "data/labeled_samples"
OUTPUT = "data/labeled_samples.json"

# 文件名 → content_type 映射
CATEGORY_MAP = {
    "game_cover":           "game_cover",
    "vocaloid_original":    "vocaloid_original",
    "vocaloid_cover":       "vocaloid_cover",
    "irrelevant":           "irrelevant",
    "other":                "other",
}

samples = []

for fname in sorted(os.listdir(SRC_DIR)):
    if not fname.endswith(".csv"):
        continue

    label = None
    for key, val in CATEGORY_MAP.items():
        if key in fname:
            label = val
            break

    if label is None:
        print(f"  跳过: {fname}（无法匹配分类）")
        continue

    path = os.path.join(SRC_DIR, fname)
    df = pd.read_csv(path)

    # 只保留 title 和 tags
    for _, row in df.iterrows():
        copyright_val = row.get("copyright", -1)
        copyright_str = "自制" if copyright_val == 1 else "转载" if copyright_val == 2 else "未知"

        samples.append({
            "title":        str(row["title"]),
            "tags":         str(row.get("tags", "")),
            "category":     str(row.get("category", "")),         # 视频分区，如 VOCALOID·UTAU
            "copyright":    copyright_str,                         # 版权标记
            "author":       str(row.get("author", "")),            # UP主（仅供参考）
            "content_type": label,
        })

    print(f"  {fname}: {len(df)} 条 → {label}")

# 打乱顺序（避免同类聚集影响训练）
import random
random.shuffle(samples)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"\n输出: {OUTPUT}（共 {len(samples)} 条）")

# 统计
from collections import Counter
dist = Counter(s["content_type"] for s in samples)
for label, count in dist.most_common():
    print(f"  {label}: {count}")
