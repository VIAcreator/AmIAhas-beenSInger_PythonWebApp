"""
模块2：数据预处理与清洗
步骤: 去重 → 分类 → 标题解析 → 通用清洗 → 衍生特征 → 过滤 → 歌曲聚合
"""

import re
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
# 步骤2：内容分类（正则兜底，QLoRA 后期替换）
# ==========================================================================

GAME_KEYWORDS = [
    "PJSK", "プロセカ", "Project Sekai", "世界计划",
    "BangDream", "バンドリ", "BanG Dream", "少女乐团派对",
    "Phigros", "Arcaea", "Deemo", "Cytus", "Cytus II",
    "CHUNITHM", "チュウニズム", "中二节奏",
    "maimai", "舞萌", "osu!",
    "D4DJ", "ラブライブ", "LoveLive", "偶像大师", "アイマス",
    "太鼓の達人", "太鼓达人", "Muse Dash", "VOEZ",
    "プロジェクトセカイ",
]


def classify_with_regex(title: str, tags: str, is_reupload: bool) -> str:
    """
    正则规则分类（兜底方案）。QLoRA 模型可用时替换此函数。

    输入:
        title:         str  — 视频/分P标题
        tags:          str  — 逗号分隔的标签
        is_reupload:   bool — 标题解析后的是否搬运标记

    输出:
        str  — "game_cover" | "vocaloid_original" | "vocaloid_cover" |
               "irrelevant" | "other"

    分类优先级: game_cover > reupload > vocaloid_original > vocaloid_cover > other
    """
    title_upper = title.upper()
    tags_upper = tags.upper()

    for kw in GAME_KEYWORDS:
        if kw.upper() in title_upper or kw.upper() in tags_upper:
            return "game_cover"

    if is_reupload:
        return "reupload"

    if re.search(r"オリジナル|原创曲|原创[曲PV]|Original", title):
        return "vocaloid_original"

    if re.search(r"カバー|Cover|翻唱|歌って[み見]た", title):
        return "vocaloid_cover"

    return "vocaloid_original"


def classify_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 DataFrame 逐行分类，新增 content_type 列。
    当前调用正则分类，后期替换为 QLoRA 推理。
    """
    df = df.copy()
    df["content_type"] = df.apply(
        lambda row: classify_with_regex(row["title"], row["tags"], row.get("is_reupload", False)),
        axis=1
    )
    return df


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
    report["content_type"] = df["content_type"].value_counts().to_dict()

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
