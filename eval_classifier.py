"""
加载训练好的 QLoRA 模型，在测试集上评测分类效果。

使用方法:
    source venv/bin/activate
    python eval_classifier.py
"""

import json
import os
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report

# 同步自 train_classifier.py
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_PATH = "models/content_classifier_lora"
TEST_PATH = "data/test_split.json"

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
    "vocaloid_cover":       "IA（作为 VOCALOID 歌姬）翻唱别人的歌曲",
    "irrelevant":           "与虚拟歌姬 IA 完全无关的内容（游戏、军事、AI生成等）",
    "other":                "与虚拟歌姬 IA 相关但非歌曲投稿（语调教、演唱会、科普教学、猜歌比赛、MMD舞蹈、榜单盘点等）",
}

SYSTEM_PROMPT = (
    "你是一个B站视频分类助手。根据视频标题、标签、分区和版权信息，将视频分为5类：\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in LABEL_DEFINITIONS.items())
    + "\n\n只输出标签名，不要解释。"
)


def main():
    import mlx_lm

    print("=" * 60)
    print("QLoRA 分类器评测")
    print("=" * 60)
    print(f"模型: {MODEL_PATH}")
    print(f"测试集: {TEST_PATH}")
    print()

    # 1. 加载测试集
    if not os.path.exists(TEST_PATH):
        print(f"错误: 测试集 {TEST_PATH} 不存在，请先运行 train_classifier.py")
        return

    with open(TEST_PATH, encoding="utf-8") as f:
        test_samples = json.load(f)
    print(f"加载 {len(test_samples)} 条测试样本")

    dist = Counter(s["content_type"] for s in test_samples)
    for label in LABELS:
        print(f"  {label:25s}: {dist.get(label, 0)}")

    # 2. 加载模型 + LoRA 适配器
    print("\n[1/2] 加载模型...")
    model, tokenizer = mlx_lm.load(
        MODEL_NAME,
        adapter_path=MODEL_PATH,
    )
    print(f"  已加载基座模型 + LoRA 适配器: {MODEL_PATH}")

    # 3. 逐条推理
    print(f"[2/2] 评测 {len(test_samples)} 条...")
    predictions = []
    true_labels = []

    for i, sample in enumerate(test_samples):
        input_text = (
            f"标题：{sample['title']}\n"
            f"标签：{sample['tags']}\n"
            f"分区：{sample.get('category', '')}\n"
            f"版权：{sample.get('copyright', '')}"
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

        predictions.append(pred)
        true_labels.append(sample["content_type"])

        if (i + 1) % 10 == 0 or i == len(test_samples) - 1:
            print(f"  {i+1}/{len(test_samples)} ...")

    # 4. 统计
    acc = accuracy_score(true_labels, predictions)
    report = classification_report(
        true_labels, predictions, labels=LABELS, zero_division=0
    )

    print(f"\n准确率: {acc:.2%}")
    print(report)

    # 5. 输出错误案例
    print("错误分类:")
    for i, (true, pred) in enumerate(zip(true_labels, predictions)):
        if true != pred:
            s = test_samples[i]
            print(f"  [{true} → {pred}] {s['title'][:60]}")


if __name__ == "__main__":
    main()
