#!/bin/bash
# 训练 + 评测（可后台运行，断开终端不中断）
# 使用方法:
#   nohup bash run_train.sh > train_log.txt 2>&1 &
#   tail -f train_log.txt  # 查看进度

set -e  # 任何命令失败则退出

export HF_ENDPOINT=https://hf-mirror.com

source venv/bin/activate

echo "========================================"
echo "QLoRA 训练开始"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "模型: Qwen2.5-1.5B-Instruct"
echo "数据: labeled_samples/*.csv (1989条)"
echo "分类: ia_music | ia_related | irrelevant"
echo "========================================"
echo ""

# 训练
echo "[1/2] 训练..."
python train_classifier.py
TRAIN_EXIT=$?
echo "训练完成 (exit=$TRAIN_EXIT)"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "训练失败，跳过评测。检查上方错误信息。"
    exit $TRAIN_EXIT
fi

echo ""
echo "========================================"
echo "[2/2] 评测..."
python eval_classifier.py
EVAL_EXIT=$?

echo ""
echo "========================================"
echo "全部完成 — $(date '+%Y-%m-%d %H:%M:%S')"
echo "训练 exit=$TRAIN_EXIT, 评测 exit=$EVAL_EXIT"
echo "模型: models/content_classifier_lora/"
echo "查看结果: tail -60 train_log.txt"
echo "========================================"
