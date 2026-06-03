"""
模块2：数据预处理与清洗
步骤: 去重 → 分类 → 标题解析 → 通用清洗 → 衍生特征 → 过滤 → 歌曲聚合
"""

import pandas as pd
from datetime import datetime
from flask import jsonify


# ==========================================================================
# 步骤1：去重（练习函数）
# ==========================================================================

def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    按 (bvid, page) 复合键去重，保留首次出现的记录。

    输入:
        df: pd.DataFrame  — 原始数据，需包含 "bvid" 和 "page" 两列

    输出:
        tuple:
          [0] pd.DataFrame  — 去重后的 DataFrame（行顺序保留）
          [1] dict           — {
                "before":  4128,    # 去重前总行数
                "after":   4100,    # 去重后行数
                "removed":   28     # 被移除的重复行数
              }
    """
    dropped_df = df.drop_duplicates(subset=["bvid", "page"], keep='first')
    before = len(df)
    after = len(dropped_df)
    return dropped_df, {"before": before, "after": after, "removed": before - after}


# ==========================================================================
# 步骤2：内容分类（正则，QLoRA 后期替换）
# ==========================================================================

from modules.content_classifier import classify_all, CONTENT_TYPES, CLASSIFICATION_CONFIG


# ==========================================================================
# 步骤3：标题解析（后续编写）
# ==========================================================================

def parse_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    从标题中解析: song_name, original_creator, vocal_singer, is_reupload。
    新增 4 列，返回新 DataFrame。
    """
    # TODO: 后续实现
    return df


# ==========================================================================
# 步骤4-5：通用清洗 + 衍生特征（后续编写）
# ==========================================================================

def general_clean(df: pd.DataFrame) -> pd.DataFrame:
    """缺失值处理 + 日期标准化 + 异常值过滤"""
    # TODO: 后续实现
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造 days_since_pub, daily_avg_plays, engagement_rate"""
    # TODO: 后续实现
    return df


# ==========================================================================
# 步骤6：过滤 + 聚合（后续编写）
# ==========================================================================

def filter_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    """移除 content_type == 'irrelevant' 的记录"""
    return df[df["content_type"] != "irrelevant"].copy()


def aggregate_to_songs(df: pd.DataFrame) -> pd.DataFrame:
    """按 (song_name, original_creator) 聚合为歌曲粒度"""
    # TODO: 后续实现
    return pd.DataFrame()


# ==========================================================================
# Flask 路由处理
# ==========================================================================

def handle_clean(raw_df, set_data_fn):
    """
    POST /api/clean 的处理函数。
    逐步执行清洗管道，返回每步报告，最终存入双层 DF。
    """
    if raw_df is None:
        return jsonify({"error": "尚未加载数据，请先上传文件"}), 400

    report = {}

    # 步骤1：去重
    df, report["dedup"] = deduplicate(raw_df)

    # 步骤2：内容分类
    df = classify_all(df)
    report["content_type_regex"] = df["content_type"].value_counts().to_dict()
    report["suspicious_count"] = int(df["suspicious"].sum())

    # 步骤2b：LLM 验证可疑条目（DeepSeek 或 QLoRA，用户可选）
    from modules.llm_verifier import verify_suspicious
    df = verify_suspicious(df)
    report["content_type_final"] = df["content_type"].value_counts().to_dict()
    report["is_game_count"] = int(df["is_game"].sum())
    report["is_cover_count"] = int(df["is_cover"].sum())

    # 步骤3：标题解析
    df = parse_titles(df)

    # 步骤4：通用清洗
    df = general_clean(df)

    # 步骤5：衍生特征
    df = add_features(df)

    # 步骤6：过滤 + 歌曲聚合
    df_clean = filter_irrelevant(df)
    report["filtered"] = {
        "before": len(df),
        "after": len(df_clean),
        "removed_irrelevant": len(df) - len(df_clean),
    }
    df_songs = aggregate_to_songs(df_clean)
    report["songs"] = {
        "total_songs": len(df_songs),
        "avg_videos_per_song": round(len(df_clean) / len(df_songs), 1) if len(df_songs) > 0 else 0,
    }

    # 存入缓存
    set_data_fn(raw_df=raw_df, clean_df=df_clean, songs_df=df_songs)

    return jsonify({"status": "ok", "report": report})
