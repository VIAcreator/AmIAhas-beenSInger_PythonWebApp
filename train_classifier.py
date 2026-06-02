"""
QLoRA 微调 B站 IA 视频内容分类器
=================================
基座模型: Qwen2.5-1.5B-Instruct
微调方法: 4-bit QLoRA (r=8, alpha=16)
输出: models/content_classifier_lora/

使用方法:
    source venv/bin/activate
    python train_classifier.py
"""

import json
import os
import random
from collections import Counter
from typing import Optional

# ==========================================================================
# 配置
# ==========================================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "data/labeled_samples.json"
OUTPUT_DIR = "models/content_classifier_lora"
TEST_SIZE = 0.15                  # 测试集比例
SEED = 42

# LoRA 参数
LORA_R = 8
LORA_ALPHA = 16
MAX_SEQ_LENGTH = 512

# 训练参数
BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 2e-4

# 五分类标签及其定义
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


# ==========================================================================
# 数据加载与划分
# ==========================================================================

def load_data(path: str) -> list[dict]:
    """加载标注样本 JSON。"""
    with open(path, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"加载 {len(samples)} 条标注样本")
    return samples


def stratified_split(samples: list[dict], test_ratio: float, seed: int
                     ) -> tuple[list[dict], list[dict]]:
    """
    分层划分训练集/测试集。每类至少保留 1 条在测试集。

    输入:
        samples:    标注样本列表
        test_ratio: 测试集比例 (0~1)
        seed:       随机种子

    输出:
        (train_samples, test_samples)
    """
    random.seed(seed)

    by_label = {}
    for s in samples:
        by_label.setdefault(s["content_type"], []).append(s)

    train, test = [], []
    for label, items in by_label.items():
        random.shuffle(items)
        n_test = max(1, int(len(items) * test_ratio))
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    random.shuffle(train)
    random.shuffle(test)

    print(f"\n划分: train={len(train)}, test={len(test)}")
    for label in LABELS:
        t_count = sum(1 for s in train if s["content_type"] == label)
        e_count = sum(1 for s in test if s["content_type"] == label)
        print(f"  {label:25s}: train={t_count:>3}, test={e_count:>2}")

    return train, test


# ==========================================================================
# 样本格式化
# ==========================================================================

def format_sample(s: dict) -> str:
    """
    将一条标注样本格式化为 Alpaca 训练文本。

    输入格式:
        {"title": "...", "tags": "...", "category": "...",
         "copyright": "...", "content_type": "..."}

    输出格式:
        ### Instruction:
        <SYSTEM_PROMPT>
        ### Input:
        标题：xxx
        标签：xxx
        分区：xxx
        版权：xxx
        ### Response:
        game_cover
    """
    input_text = (
        f"标题：{s['title']}\n"
        f"标签：{s['tags']}\n"
        f"分区：{s.get('category', '')}\n"
        f"版权：{s.get('copyright', '')}"
    )
    return (
        f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n{s['content_type']}"
    )


# ==========================================================================
# 类别权重计算
# ==========================================================================

def compute_class_weights(train_samples: list[dict]) -> dict[str, float]:
    """
    根据训练集样本数计算反比权重，缓解类别不平衡。

    公式: weight = total_samples / (n_classes × class_count)

    输出: {"game_cover": 2.5, "vocaloid_original": 0.6, ...}
    """
    counter = Counter(s["content_type"] for s in train_samples)
    total = len(train_samples)
    n = len(LABELS)
    weights = {}
    for label in LABELS:
        count = counter.get(label, 1)
        weights[label] = total / (n * count)
    print("\n类别权重:")
    for label in LABELS:
        print(f"  {label:25s}: {counter.get(label, 0):>3} 条 → 权重 {weights[label]:.2f}")
    return weights


# ==========================================================================
# 训练
# ==========================================================================

def train(train_samples: list[dict], test_samples: list[dict],
          class_weights: dict[str, float]):
    """
    QLoRA 微调主流程。
    """
    from unsloth import FastLanguageModel, MLXTrainer, MLXTrainingConfig
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report

    # --- 第1步：加载 4-bit 量化模型 ---
    print("\n[1/6] 加载模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # --- 第2步：添加 LoRA 适配器 ---
    print("[2/6] 添加 LoRA 适配器...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
    )

    # --- 第3步：格式化数据 ---
    print("[3/6] 格式化训练数据...")
    train_texts = [format_sample(s) for s in train_samples]
    # MLXTrainer 接受 list[dict] 格式，每项需含 text 字段
    train_dataset = [{"text": t} for t in train_texts]

    # --- 第4步：MLX 训练 ---
    print("[4/6] 开始训练...")
    config = MLXTrainingConfig(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        max_steps=-1,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        output_dir=OUTPUT_DIR,
        report_to="none",
        max_seq_length=MAX_SEQ_LENGTH,
    )
    trainer = MLXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=config,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )
    trainer.train()

    # --- 第5步：评估 ---
    print("\n[5/6] 评估测试集...")
    predictions = []
    true_labels = []

    for sample in test_samples:
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

        import mlx_lm
        result = mlx_lm.generate(
            model, tokenizer, prompt=prompt, max_tokens=10, verbose=False
        )

        # 从生成结果中提取标签（最后一个有效标签）
        pred = "other"
        for label in LABELS:
            if label in result.lower():
                pred = label
        predictions.append(pred)
        true_labels.append(sample["content_type"])

    acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, labels=LABELS, zero_division=0)

    print(f"  准确率: {acc:.2%}")
    print(report)

    # --- 第6步：保存适配器 ---
    print("[6/6] 保存模型...")
    model.save_pretrained_gguf(OUTPUT_DIR, tokenizer)

    # 保存评测报告
    with open(os.path.join(OUTPUT_DIR, "eval_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.2%}\n\n")
        f.write(report)


# ==========================================================================
# 主流程
# ==========================================================================

def main():
    print("=" * 60)
    print("QLoRA 内容分类器训练")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"数据: {DATA_PATH}")
    print(f"输出: {OUTPUT_DIR}")
    print()

    # 1. 加载数据
    samples = load_data(DATA_PATH)
    if not samples:
        print("无标注数据，请先运行 crawler/build_labeled_samples.py")
        return

    # 检查必要字段
    required_cols = {"title", "content_type"}
    for s in samples:
        missing = required_cols - set(s.keys())
        if missing:
            print(f"错误: 样本缺少字段 {missing}")
            print(f"  样本: {s}")
            return

    # 统计分布
    dist = Counter(s["content_type"] for s in samples)
    print("标注分布:")
    for label in LABELS:
        print(f"  {label:25s}: {dist.get(label, 0)}")

    # 2. 划分训练/测试集
    train_samples, test_samples = stratified_split(samples, TEST_SIZE, SEED)

    # 3. 计算类别权重
    class_weights = compute_class_weights(train_samples)

    # 4. 训练
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train(train_samples, test_samples, class_weights)

    print(f"\n完成！模型已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
