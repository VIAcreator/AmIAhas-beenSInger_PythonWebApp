"""
在全部 4128 条数据上推理模型，输出分类分布和样本抽查。

使用方法:
    source venv/bin/activate
    python eval_full_dataset.py
"""

import json
import os
import pandas as pd
from collections import Counter
import mlx_lm

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_PATH = "models/content_classifier_lora"
CSV_PATH = "data/ia_music_data.csv"
OUTPUT = "data/full_classification.csv"
SAMPLE_SIZE = 100  # 随机抽查行数（全量 4128 条全部推理）

LABELS = ["ia_music", "ia_related", "irrelevant"]

LABEL_DEFINITIONS = {
    "ia_music": (
        "IA（作为 VOCALOID/CeVIO 歌姬）演唱的音乐作品。"
        "包括原创、翻唱、音游曲、钢琴改编、合唱等"
    ),
    "ia_related": (
        "与虚拟歌姬 IA 相关但非歌曲投稿。"
        "语调教、演唱会、科普/P主人物志、猜歌比赛、MMD舞蹈、榜单盘点、声库评测等"
    ),
    "irrelevant": (
        "与虚拟歌姬 IA 完全无关。"
        "游戏实况、军事/飞机型号(IA58)、AI生成内容、漫剧、音响器材等"
    ),
}

SYSTEM_PROMPT = (
    "你是一个B站视频分类助手。根据视频标题、标签、分区信息，将视频分为3类：\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in LABEL_DEFINITIONS.items())
    + "\n\n只输出标签名，不要解释。"
)


def classify_one(model, tokenizer, title, tags, category, copyright_str):
    input_text = (
        f"标题：{title}\n"
        f"标签：{tags}\n"
        f"分区：{category}\n"
        f"版权：{copyright_str}"
    )
    prompt = (
        f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
        f"### Input:\n{input_text}\n\n### Response:\n"
    )
    result = mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=10, verbose=False
    )
    pred = "other"
    result_lower = result.lower()
    for label in LABELS:
        if label in result_lower:
            pred = label
            break
    return pred


def main():
    # 1. 加载数据
    print("[1/3] 加载数据...")
    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates(subset=["bvid", "page"])
    print(f"  去重后: {len(df)} 行")

    # 2. 加载模型
    print("[2/3] 加载模型...")
    model, tokenizer = mlx_lm.load(MODEL_NAME, adapter_path=MODEL_PATH)

    # 3. 全量推理
    print(f"[3/3] 推理 {len(df)} 条...")
    predictions = []

    for i, (_, row) in enumerate(df.iterrows()):
        copyright_val = row.get("copyright", -1)
        copyright_str = (
            "自制" if copyright_val == 1
            else "转载" if copyright_val == 2
            else "未知"
        )
        pred = classify_one(
            model, tokenizer,
            title=str(row["title"]),
            tags=str(row.get("tags", "")),
            category=str(row.get("category", "")),
            copyright_str=copyright_str,
        )
        predictions.append(pred)

        if (i + 1) % 100 == 0:
            dist = Counter(predictions)
            print(f"  {i+1}/{len(df)} | "
                  + " | ".join(f"{l.split('_')[-1]}:{dist.get(l,0)}" for l in LABELS))

    # 4. 统计
    df["content_type_llm"] = predictions
    dist = Counter(predictions)

    print(f"\n分类分布 ({len(df)} 条):")
    for label in LABELS:
        count = dist.get(label, 0)
        pct = count / len(df) * 100
        print(f"  {label:25s}: {count:>5} ({pct:5.1f}%)")

    # 5. 保存
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {OUTPUT}")

    # 6. 每类抽查 3 条
    print("\n=== 每类抽查 3 条 ===")
    for label in LABELS:
        subset = df[df["content_type_llm"] == label]
        print(f"\n--- {label} ({len(subset)}条) ---")
        for _, r in subset.sample(min(3, len(subset))).iterrows():
            print(f"  [{r['author']}] {r['title'][:55]}")


if __name__ == "__main__":
    main()
