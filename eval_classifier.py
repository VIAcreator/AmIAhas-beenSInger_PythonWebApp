"""
加载训练好的 QLoRA 模型，在测试集上评测。

使用方法:
    source venv/bin/activate
    python eval_classifier.py
"""

import json
import os
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_PATH = "models/content_classifier_lora"
TEST_PATH = "data/test_split.json"

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
    "你是一个B站视频分类助手。根据视频标题、标签、分区和版权信息，将视频分为3类：\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in LABEL_DEFINITIONS.items())
    + "\n\n只输出标签名，不要解释。"
)


def main():
    import mlx_lm

    print("=" * 60)
    print("QLoRA 分类器评测")
    print("=" * 60)

    if not os.path.exists(TEST_PATH):
        print(f"测试集 {TEST_PATH} 不存在，请先运行 train_classifier.py")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"模型 {MODEL_PATH} 不存在，请先运行 train_classifier.py")
        return

    with open(TEST_PATH, encoding="utf-8") as f:
        test_samples = json.load(f)
    print(f"测试集: {len(test_samples)} 条")
    dist = Counter(s["content_type"] for s in test_samples)
    for label in LABELS:
        print(f"  {label}: {dist.get(label, 0)}")

    print("\n加载模型...")
    model, tokenizer = mlx_lm.load(MODEL_NAME, adapter_path=MODEL_PATH)

    print(f"推理 {len(test_samples)} 条...")
    predictions, true_labels = [], []

    for i, s in enumerate(test_samples):
        input_text = (
            f"标题：{s['title']}\n"
            f"标签：{s.get('tags', '')}\n"
            f"分区：{s.get('category', '')}"
        )
        prompt = (
            f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
            f"### Input:\n{input_text}\n\n### Response:\n"
        )
        result = mlx_lm.generate(
            model, tokenizer, prompt=prompt, max_tokens=5, verbose=False
        )
        pred = "ia_music"
        for label in LABELS:
            if label in result.lower():
                pred = label
                break
        predictions.append(pred)
        true_labels.append(s["content_type"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_samples)}")

    acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, labels=LABELS, zero_division=0)
    print(f"\n准确率: {acc:.2%}")
    print(report)

    # 错误案例
    errors = [(s, t, p) for s, t, p in zip(test_samples, true_labels, predictions) if t != p]
    print(f"错误案例 ({len(errors)} 条):")
    for s, true, pred in errors:
        print(f"  [{true} -> {pred}] {s['title'][:70]}")


if __name__ == "__main__":
    main()
