#!/bin/bash
# 一键训练 + 评测，输出日志到 train_log.txt
# 使用方法: bash run_train.sh

export HF_ENDPOINT=https://hf-mirror.com

source venv/bin/activate

echo "========================================"
echo "开始训练 — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 训练
python train_classifier.py
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo ""
    echo "训练失败 (exit code=$TRAIN_EXIT)，跳过评测。"
    exit $TRAIN_EXIT
fi

echo ""
echo "========================================"
echo "训练完成，开始评测 — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 评测
python eval_classifier.py
EVAL_EXIT=$?

echo ""
echo "========================================"
echo "全部完成 — $(date '+%Y-%m-%d %H:%M:%S')"
echo "训练 exit=$TRAIN_EXIT, 评测 exit=$EVAL_EXIT"
echo "模型: models/content_classifier_lora/"
echo "========================================"
