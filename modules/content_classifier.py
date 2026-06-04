"""
模块2子模块：内容分类器（正则版）
基于 1,989 条标注样本迭代优化，准确率 94.3%。
LLM 可用时替换 classify() 为 QLoRA 推理。
"""

import re
import pandas as pd


# ==========================================================================
# 分类标签定义
# ==========================================================================

CONTENT_TYPES = ["ia_music", "ia_related", "irrelevant"]

CLASSIFICATION_CONFIG = {
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

# ==========================================================================
# 规则关键词（基于标注数据分析）
# ==========================================================================

GAME_KEYWORDS = [
    "PJSK", "プロセカ", "Project Sekai", "世界计划", "プロジェクトセカイ",
    "BangDream", "バンドリ", "BanG Dream", "少女乐团派对",
    "Phigros", "Arcaea", "Deemo", "Cytus", "Cytus II",
    "CHUNITHM", "チュウニズム", "中二节奏",
    "maimai", "舞萌", "osu!", "D4DJ",
    "Muse Dash", "VOEZ", "太鼓达人", "太鼓の達人",
    "音乐游戏", "音游",
]

COVER_KEYWORDS = [
    "Cover", "COVER", "cover",
    "カバー",
    "翻唱", "翻弹", "翻调", "合唱",
    "歌ってみた", "歌って見た", "歌ってみ",
    "COVERED", "Covered",
    "日语版", "中文版", "英語版", "英语版", "韓国語版",
    "REMIX", "Remix", "remix",
    "演奏", "钢琴", "吉他", "电吉他", "混音", "Piano",
    "试着唱", "试着跳", "试着演奏", "试着做了",
    "入坑曲重调",
    "声真似", "模仿",
]

VOCALOID_TAGS = [
    "VOCALOID", "vocaloid", "VOCALOID殿堂入り", "VOCALOID3",
    "CeVIO", "CeVIO AI", "UTAU", "ボカロ", "ボーカロイド",
    "V家", "術力口", "术力口", "虚拟歌姬", "虚拟歌手", "虚拟偶像",
]

SINGER_TAGS = [
    "初音ミク", "初音未来", "初音MIKU", "Miku", "MIKU",
    "GUMI", "鏡音リン", "鏡音レン", "镜音RIN", "镜音LEN",
    "巡音ルカ", "巡音LUKA", "MEIKO", "KAITO", "MAYU",
    "結月ゆかり", "结月缘", "Fukase", "Flower", "v_flower",
    "可不", "重音テト", "重音TETO", "ONE", "OИE",
    "IA小天使", "IA ROCKS", "IA GLOWB",
    "洛天依", "言和", "乐正绫", "IA:[R]", "miki",
]

NON_SONG_KEYWORDS = [
    "人物志", "科普", "编年史", "发生了什么", "小课堂",
    "盘点", "周刊", "RANKING", "ランキング", "推荐",
    "纪录片", "官方频", "官方OFFICIAL", "OFFICIAL",
    "在家动手", "猜歌", "猜曲", "猜术曲", "介绍", "小剧场", "同居",
    "实验", "靶场", "样本试听", "声音样本",
    "手书", "手描", "描改", "绘画", "编舞", "跳舞",
    "演唱会", "巡演", "现场", "超Party", "超PARTY",
    "来了", "新宝岛",
    "画师", "COS", "Cosplay",
    "剧场", "VOICEROID", "VOICEROID剧场",
    "语调教", "跨语种",
    "声库", "设计", "简介", "这到底是什么",
    "娇喘", "信念感",
    "组曲", "听力", "请问",
    "教程", "教学", "入门",
    "PK", "对决", "哪个", "比较",
    "P主推荐", "P主人物", "P主音乐电台", "电台",
    "授权汉化",
    "宣布", "发表",
    "不幸", "自杀", "去世",
    "新闻", "速报",
    "报告", "统计", "评论", "WOTA艺",
    "排行", "TOP100", "TOP50", "BEST", "传送", "精选",
    "超过", "为止",
    "贺岁", "拜年", "单品", "原创动画", "人气",
    "传说曲", "殿堂", "神话曲", "名曲",
    "调查", "问卷", "投票", "第一期", "第二期", "第三期",
    "试着跳", "试着做了", "试着演奏", "试着画",
]

HIGH_PRIORITY_IRRELEVANT = [
    "Roblox", "roblox", "物品避难所", "物品庇护所",
    "Item Asylum", "item asylum", "itemasylum", "ItemAsylum",
    "千问", "三角洲", "枪花", "漫剧",
]

IRRELEVANT_SIGNALS = [
    "战争雷霆", "坦克", "攻击机", "战斗机", "军舰",
    "AI生成", "恐龙快打", "I want to eat",
    "装甲核心", "ACVI",
    "Apex", "APEX", "CSGO", "Valorant",
    "赛博朋克", "巫师3",
    "内鬼", "PVZ", "我的世界", "饥荒",
    "梗图", "meme", "搞笑配音",
    "荒野乱斗", "Brawl",
    "绝区零", "鸣潮",
    "假面骑士", "奥特曼",
    "星际争霸", "崩坏", "丽磁",
    "绝区零UP主激励计划", "鸣潮创作激励计划", "激励计划", "kpop",
]

MUSIC_CATEGORIES = [
    "VOCALOID·UTAU", "翻唱", "演奏", "音乐综合",
    "音游", "MV", "音MAD", "原创音乐", "音乐现场", "音乐教学",
]


# ==========================================================================
# 单条分类
# ==========================================================================

def classify(title: str, tags: str, category: str) -> dict:
    """
    正则分类单条记录。

    输入:
        title:    str  — 视频/分P标题
        tags:     str  — 逗号分隔的标签字符串
        category: str  — B站分区名如 "VOCALOID·UTAU"

    输出:
        dict  — {
            "content_type":       "ia_music" | "ia_related" | "irrelevant",
            "is_game":            bool,
            "is_cover":           bool,
            "rule":               str,     # 命中的规则名
            "suspicious":         bool,    # 是否需要 LLM 复核
            "suspicious_reason":  str,     # 可疑原因
        }
    """
    t_upper = title.upper() if title else ""
    g_upper = tags.upper() if tags else ""
    tags_list = [x.strip() for x in tags.split(",")] if tags else []

    # ---- 布尔标记 ----
    is_game = False
    for kw in GAME_KEYWORDS:
        if kw.upper() in t_upper or kw.upper() in g_upper:
            is_game = True
            break

    is_cover = False
    for kw in COVER_KEYWORDS:
        if kw.upper() in t_upper or kw.upper() in g_upper:
            is_cover = True
            break

    # ---- 预处理信号 ----
    has_vocaloid_tag = any(vt.upper() in g_upper for vt in VOCALOID_TAGS)
    has_singer_tag = any(st.upper() in g_upper for st in SINGER_TAGS)

    has_ia_exact = False
    for t in tags_list:
        tu = t.upper()
        if tu == "IA" or tu.startswith("IA ") or "IA_" in tu or tu.endswith(" IA"):
            has_ia_exact = True
            break
    has_ia_loose = has_ia_exact or any("IA" in t.upper() for t in tags_list)

    music_signal = has_vocaloid_tag or has_singer_tag or has_ia_exact
    any_ia = has_ia_loose or has_vocaloid_tag or has_singer_tag
    only_ia_signal = has_ia_exact and not has_vocaloid_tag and not has_singer_tag

    in_music_category = any(
        mc.upper() in (category or "").upper() for mc in MUSIC_CATEGORIES
    )

    strong_title_music = bool(
        re.search(r'feat\.?\s*IA', title, re.I) or
        re.search(r'【IA[^】]*】', title) or
        re.search(r'[/／][^/／]+feat', title, re.I)
    )
    no_ia_format = not strong_title_music

    clean_title = re.sub(r'【[^】]*】', '', title).strip()
    clean_title = re.sub(r'\[[^\]]*\]', '', clean_title).strip()
    weird_title = (
        len(clean_title) < 8 or
        "?" in title or "？" in title or
        "..." in title or "!!!" in title
    )

    # ---- 分类判定 ----
    result = {
        "is_game": is_game, "is_cover": is_cover,
        "rule": "", "suspicious": False, "suspicious_reason": "",
    }

    # 1. 最高优先级：强无关词 → 直接 irrelevant
    for kw in HIGH_PRIORITY_IRRELEVANT:
        if kw.upper() in t_upper or kw.upper() in g_upper:
            result.update({"content_type": "irrelevant", "rule": f"high_priority:{kw}"})
            return result

    # 2. 完全没信号 → irrelevant
    if not any_ia and not strong_title_music:
        cat_upper = (category or "").upper()
        music_cats = ["VOCALOID", "UTAU", "音MAD", "翻唱", "演奏", "音乐"]
        if not any(mc in cat_upper for mc in music_cats):
            result.update({"content_type": "irrelevant", "rule": "no_signal"})
            return result

    # 3. 标题含无关信号 → irrelevant
    if not strong_title_music:
        for kw in IRRELEVANT_SIGNALS:
            if kw.upper() in t_upper:
                result.update({"content_type": "irrelevant", "rule": f"irrelevant_kw:{kw}"})
                return result

    # 4. 标题含 non-song 关键词
    hit_nonsong = None
    if not strong_title_music:
        for kw in NON_SONG_KEYWORDS:
            if kw.upper() in t_upper:
                hit_nonsong = kw
                break

    if hit_nonsong:
        if music_signal:
            result.update({
                "content_type": "ia_related", "rule": f"nonsong_mixed:{hit_nonsong}",
                "suspicious": True,
                "suspicious_reason": f"标题含'{hit_nonsong}'(非歌曲信号)但tags有音乐标签，需LLM确认",
            })
            return result
        else:
            result.update({"content_type": "ia_related", "rule": f"nonsong:{hit_nonsong}"})
            return result

    # 5. 有音乐信号但标题无【IA】格式
    if music_signal and no_ia_format:
        if only_ia_signal:
            result.update({
                "content_type": "ia_music", "rule": "music_only_ia",
                "suspicious": True,
                "suspicious_reason": (
                    f"tags仅有'IA'但无VOCALOID/歌姬标签，"
                    f"IA可能是笔误或非虚拟歌姬(clean='{clean_title[:30]}')，需LLM确认"
                ),
            })
            return result
        if weird_title:
            result.update({
                "content_type": "ia_music", "rule": "music_weird_title",
                "suspicious": True,
                "suspicious_reason": (
                    f"tags有IA/VOCALOID但标题格式非标准歌曲"
                    f"(clean='{clean_title[:30]}')，需LLM确认"
                ),
            })
            return result
        result.update({"content_type": "ia_music", "rule": "music_signal"})
        return result

    # 6. 有音乐信号 + 标题明确 → 高置信度 ia_music
    if music_signal or strong_title_music:
        result.update({"content_type": "ia_music", "rule": "music_signal"})
        return result

    # 7. 兜底：category 推断
    cat_upper = (category or "").upper()
    if "VOCALOID" in cat_upper or "UTAU" in cat_upper or "翻唱" in cat_upper:
        result.update({
            "content_type": "ia_music", "rule": "category_vocaloid",
            "suspicious": True,
            "suspicious_reason": f"无明确信号，仅靠分区'{category}'推断为ia_music",
        })
        return result
    if "MMD" in cat_upper or "手书" in cat_upper:
        result.update({
            "content_type": "ia_related", "rule": "category_mmd",
            "suspicious": True,
            "suspicious_reason": f"无明确信号，仅靠分区'{category}'推断",
        })
        return result
    if "游戏" in cat_upper or "日常" in cat_upper or "数码" in cat_upper:
        result.update({
            "content_type": "irrelevant", "rule": "category_gaming",
            "suspicious": True,
            "suspicious_reason": f"无明确信号，仅靠分区'{category}'推断",
        })
        return result

    result.update({
        "content_type": "ia_related", "rule": "fallback",
        "suspicious": True,
        "suspicious_reason": "无任何明确信号，默认归为ia_related",
    })
    return result


# ==========================================================================
# 批量分类 + 后处理
# ==========================================================================

def classify_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 DataFrame 逐行分类，新增 content_type/is_game/is_cover/rule/suspicious/suspicious_reason 列。

    输入:
        df: pd.DataFrame — 原始数据，需含 title, tags, category 列

    输出:
        pd.DataFrame — 原数据 + 6 列分类结果
    """
    results = []
    for _, row in df.iterrows():
        r = classify(
            str(row.get("title", "")),
            str(row.get("tags", "")),
            str(row.get("category", "")),
        )
        results.append(r)

    df = df.copy()
    df["content_type"]       = [r["content_type"] for r in results]
    df["is_game"]            = [r["is_game"] for r in results]
    df["is_cover"]           = [r["is_cover"] for r in results]
    df["rule"]               = [r["rule"] for r in results]
    df["suspicious"]         = [r["suspicious"] for r in results]
    df["suspicious_reason"]  = [r["suspicious_reason"] for r in results]

    # 后处理：ia_music 但不在音乐分区 → 标记可疑
    for pos, idx in enumerate(df.index):
        r = results[pos]
        if r["content_type"] == "ia_music" and not r["suspicious"]:
            cat = str(df.loc[idx, "category"])
            if not any(mc.upper() in cat.upper() for mc in MUSIC_CATEGORIES):
                df.at[idx, "suspicious"] = True
                df.at[idx, "suspicious_reason"] = (
                    f"分类为ia_music但分区'{cat}'非音乐分区，需LLM确认"
                )

    return df
