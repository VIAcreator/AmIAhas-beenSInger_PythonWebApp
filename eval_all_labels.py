"""
在全部 260 条标注数据上评测模型（带 prompt 缓存优化）。

使用方法:
    source venv/bin/activate
    python eval_all_labels.py
"""

import json
import os
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report
import mlx_lm

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_PATH = "models/content_classifier_lora"
DATA_PATH = "data/labeled_samples.json"

LABELS = [
    "game_cover",
    "vocaloid_original",
    "vocaloid_cover",
    "irrelevant",
    "other",
]

LABEL_DEFINITIONS = {
    "game_cover":           "音游曲翻唱/相关，如 PJSK/BangDream/Phigros/Arcaea/CHUNITHM/maimai/osu!/D4DJ 等",
    "vocaloid_original":    "VOCALOID/CeVIO 引擎合成的 IA 原创歌曲",
    "vocaloid_cover":       "IA（作为 VOCALOID 歌姬）翻唱原唱不是IA的歌曲",
    "irrelevant":           "与虚拟歌姬 IA 完全无关的内容（游戏、军事、AI生成等）",
    "other":                "与虚拟歌姬 IA 相关但非歌曲投稿（语调教、演唱会、科普教学、猜歌比赛、MMD舞蹈、榜单盘点、P主介绍等）",
}

SYSTEM_PROMPT = (
    "你是一个B站视频分类助手。根据视频标题、标签、分区和版权信息，将视频分为5类：\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in LABEL_DEFINITIONS.items())
    + "\n\n只输出标签名，不要解释。"
)

# 预构建固定前缀（所有 prompt 共用，避免重复拼接）
PREFIX = f"### Instruction:\n{SYSTEM_PROMPT}\n\n### Input:\n"


def classify_one(model, tokenizer, sample):
    """单条推理，复用预构建前缀"""
    input_text = (
        f"标题：{sample['title']}\n"
        f"标签：{sample['tags']}\n"
        f"分区：{sample.get('category', '')}\n"
        f"版权：{sample.get('copyright', '')}"
    )
    prompt = PREFIX + input_text + "\n\n### Response:\n"

    result = mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=5, verbose=False
    )
    result_lower = result.lower()
    for label in LABELS:
        if label in result_lower:
            return label
    return "other"


def main():
    # 1. 加载数据
    with open(DATA_PATH, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"标注样本: {len(samples)} 条")

    # 2. 加载模型
    print("加载模型...")
    model, tokenizer = mlx_lm.load(MODEL_NAME, adapter_path=MODEL_PATH)

    # 3. 逐条推理（max_tokens=5，预构建前缀）
    print(f"推理 {len(samples)} 条...")
    predictions = []
    true_labels = []

    for i, s in enumerate(samples):
        pred = classify_one(model, tokenizer, s)
        predictions.append(pred)
        true_labels.append(s["content_type"])

        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(samples)}")

    # 4. 统计
    acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, labels=LABELS, zero_division=0)

    print(f"\n准确率: {acc:.2%}")
    print(report)

    # 5. 混淆矩阵
    print("\n混淆矩阵 (行=真实, 列=预测):")
    header = f"{'':>25s}" + "".join(f"{l.split('_')[-1][:6]:>8s}" for l in LABELS)
    print(header)
    for i, true_label in enumerate(LABELS):
        row_counts = []
        for pred_label in LABELS:
            count = sum(1 for t, p in zip(true_labels, predictions)
                       if t == true_label and p == pred_label)
            row_counts.append(str(count))
        row = f"{true_label:>25s}" + "".join(f"{c:>8s}" for c in row_counts)
        print(row)

    # 6. 错误案例
    errors = [(s, t, p) for s, t, p in zip(samples, true_labels, predictions) if t != p]
    print(f"\n错误案例 ({len(errors)} 条):")
    for s, true, pred in errors:
        print(f"  [{true} → {pred}] {s['title'][:70]}")


if __name__ == "__main__":
    main()
