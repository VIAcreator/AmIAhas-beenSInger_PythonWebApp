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
# 步骤2：内容分类（正则 + LLM 兜底）
# ==========================================================================

# 共享分类定义（正则和 LLM 使用同一份，见 CLASSIFICATION_CONFIG）
CONTENT_TYPES = ["game_cover", "vocaloid_original", "vocaloid_cover", "irrelevant", "other"]

CLASSIFICATION_CONFIG = {
    "game_cover": (
        "音游曲翻唱/相关。标题或tags中出现音游名称（PJSK/プロセカ/BangDream/バンドリ/"
        "Phigros/Arcaea/CHUNITHM/maimai/osu!/D4DJ/太鼓达人/Muse Dash等）"
    ),
    "vocaloid_original": (
        "VOCALOID/CeVIO 引擎合成的 IA 原创歌曲。标题含'オリジナル''原创曲''Original'"
        "等原创标记，或标题含【IA】+歌名格式且无翻唱标记"
    ),
    "vocaloid_cover": (
        "IA（作为 VOCALOID 歌姬）翻唱别人的歌曲。标题或tags含'Cover''カバー'"
        "'翻唱''歌ってみた'等翻唱标记"
    ),
    "irrelevant": (
        "与虚拟歌姬 IA 完全无关。游戏实况、军事/飞机型号(IA58)、AI生成非IA歌曲、"
        "管弦乐演奏(无IA)、标题含IA但实为其他含义"
    ),
    "other": (
        "与虚拟歌姬 IA 相关但非歌曲投稿。语调教、演唱会录像、科普教学/P主人物志、"
        "猜歌比赛、MMD舞蹈、榜单盘点、声库评测、VOICEROID剧场、纪录片"
    ),
}

# ---- 正则关键词 ----

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

# 高频音乐 signal tags（出现这些 tag 说明确实是 IA 音乐相关，加强原创判断置信度）
MUSIC_SIGNAL_TAGS = [
    "VOCALOID", "VOCALOID殿堂入り", "VOCALOID3", "vocaloid",
    "CeVIO", "CeVIO AI", "UTAU", "ボカロ", "ボーカロイド",
    "初音ミク", "GUMI", "鏡音リン", "鏡音レン", "巡音ルカ",
    "結月ゆかり", "MEIKO", "KAITO", "MAYU", "IA ROCKS",
    "IAオリジナル曲", "オリジナル曲", "VOCAROCK", "V家",
    "術力口", "术力口", "虚拟歌姬", "虚拟歌手",
]


def classify_with_regex(title: str, tags: str, is_reupload: bool) -> str | None:
    """
    正则规则分类。能确定的直接返回，不确定的返回 None 交给 LLM。

    输入:
        title:         str  — 视频/分P标题
        tags:          str  — 逗号分隔的标签
        is_reupload:   bool — 标题解析后的是否搬运标记（仅作参考，不单独成类）

    输出:
        str | None  — CONTENT_TYPES 之一，或 None（正则不确定，需 LLM 分类）

    优先级: game_cover > vocaloid_cover > vocaloid_original > None
    """
    title_upper = title.upper()
    tags_upper = tags.upper()

    # 1. 音游检测（title 或 tags 命中）
    for kw in GAME_KEYWORDS:
        if kw.upper() in title_upper or kw.upper() in tags_upper:
            return "game_cover"

    # 2. 翻唱检测（title 或 tags 命中 —— 改进2：同时检查 tags）
    cover_pattern = r"カバー|Cover|翻唱|歌って[み見]た"
    if re.search(cover_pattern, title) or re.search(cover_pattern, tags):
        return "vocaloid_cover"

    # 3. 原创检测（title 明确标注）
    if re.search(r"オリジナル|原创曲|原创[曲PV]|Original", title):
        return "vocaloid_original"

    # 4. 反向验证（改进3）: title 没明确标注，但 tags 中有音乐信号 → 很可能是原创
    for sig in MUSIC_SIGNAL_TAGS:
        if sig.upper() in tags_upper:
            return "vocaloid_original"

    # 5. 正则无法确定 → 交给 LLM
    return None


def classify_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 DataFrame 逐行分类：先用正则，正则不确定的标记为 pending。
    调用方（handle_clean）后续用 LLM 批量处理 pending 行。
    """
    df = df.copy()
    df["content_type"] = df.apply(
        lambda row: classify_with_regex(
            row["title"], row["tags"], row.get("is_reupload", False)
        ),
        axis=1,
    )
    # 正则不确定的标记为 pending
    df["content_type"] = df["content_type"].fillna("pending")
    regex_done = (df["content_type"] != "pending").sum()
    print(f"  正则分类: {regex_done}/{len(df)} 条 ({regex_done/len(df)*100:.0f}%)")
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
