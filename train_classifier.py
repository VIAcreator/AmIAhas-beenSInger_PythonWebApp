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
DATA_DIR = "data/labeled_samples"
OUTPUT_DIR = "models/content_classifier_lora"
TEST_SIZE = 0.15
SEED = 42

LORA_R = 8
LORA_ALPHA = 16
MAX_SEQ_LENGTH = 512

BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 2e-4

# v2 三分类（与 content_classifier.py 保持一致）
LABELS = ["ia_music", "ia_related", "irrelevant"]

# 标签定义（与 content_classifier.CLASSIFICATION_CONFIG 一致）
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


# ==========================================================================
# 数据加载与划分
# ==========================================================================

def load_data() -> list[dict]:
    """从 labeled_samples/ 的 3 个 CSV 加载标注数据。"""
    import csv
    FILE_LABELS = {
        "music.csv":       "ia_music",
        "irrelevent.csv":  "irrelevant",
        "related.csv":     "ia_related",
    }
    samples = []
    for fname, default_label in FILE_LABELS.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  跳过: {path} (不存在)")
            continue
        with open(path, encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            ti = header.index("title")
            gi = header.index("tags")
            ci = header.index("category") if "category" in header else -1
            li = header.index("content_type") if "content_type" in header else -1
            for row in reader:
                if len(row) <= max(ti, gi):
                    continue
                label = row[li].strip() if li >= 0 and li < len(row) else default_label
                if label not in LABELS:
                    label = default_label
                samples.append({
                    "title":    row[ti] if ti < len(row) else "",
                    "tags":     row[gi] if gi < len(row) else "",
                    "category": row[ci] if ci >= 0 and ci < len(row) else "",
                    "content_type": label,
                })
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
    input_text = (
        f"标题：{s['title']}\n"
        f"标签：{s.get('tags', '')}\n"
        f"分区：{s.get('category', '')}"
    )
    return (
        f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n{s['content_type']}"
    )


# ==========================================================================
# 上采样平衡
# ==========================================================================

def balance_by_upsampling(train_samples: list[dict]) -> list[dict]:
    """
    上采样少数类，使各类样本数接近最大值，缓解类别不平衡。

    输入: 原始训练样本列表
    输出: 上采样后的训练样本列表（顺序随机打乱）
    """
    by_label = {}
    for s in train_samples:
        by_label.setdefault(s["content_type"], []).append(s)

    max_count = max(len(items) for items in by_label.values())
    balanced = []

    for label in LABELS:
        items = by_label.get(label, [])
        if not items:
            continue
        # 复制直到数量接近 max_count
        repeats = max_count // len(items)
        remainder = max_count % len(items)
        expanded = items * repeats + random.sample(items, remainder)
        balanced.extend(expanded)

    random.shuffle(balanced)

    print("\n上采样平衡:")
    for label in LABELS:
        count = sum(1 for s in balanced if s["content_type"] == label)
        print(f"  {label:25s}: {count:>3} 条")
    return balanced


# ==========================================================================
# 训练
# ==========================================================================

def train(train_samples: list[dict]):
    """
    QLoRA 微调 + 保存。评测由 eval_classifier.py 独立完成。
    """
    from unsloth import FastLanguageModel, MLXTrainer, MLXTrainingConfig

    # --- 第1步：加载 4-bit 量化模型 ---
    print("\n[1/4] 加载模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # --- 第2步：添加 LoRA 适配器 ---
    print("[2/4] 添加 LoRA 适配器...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
    )

    # --- 第3步：格式化 + 训练 ---
    print("[3/4] 格式化 + 训练...")
    train_texts = [format_sample(s) for s in train_samples]
    train_dataset = [{"text": t} for t in train_texts]

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

    # --- 第4步：保存 ---
    print("\n[4/4] 保存模型...")
    model.save_lora_adapters(OUTPUT_DIR)
    print(f"  已保存到 {OUTPUT_DIR}")

    return model, tokenizer


# ==========================================================================
# 主流程
# ==========================================================================

def main():
    print("=" * 60)
    print("QLoRA 内容分类器训练")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"数据: {DATA_DIR}/*.csv")
    print(f"输出: {OUTPUT_DIR}")
    print()

    # 1. 加载数据
    samples = load_data()
    if not samples:
        print("无标注数据，请检查 data/labeled_samples/")
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

    # 2. 划分训练/测试集（测试集用于独立评测脚本）
    train_samples, test_samples = stratified_split(samples, TEST_SIZE, SEED)
    # 保存测试集供 eval_classifier.py 使用
    with open(os.path.join(DATA_DIR, "test_split.json"), "w", encoding="utf-8") as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)
    print(f"  测试集已保存: {len(test_samples)} 条 → data/test_split.json\n")

    # 3. 上采样平衡
    train_balanced = balance_by_upsampling(train_samples)

    # 4. 训练 + 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train(train_balanced)

    print(f"\n完成！模型已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
