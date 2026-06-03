"""
正则分类测试程序 (v2 三分类标准)
=================================
基于 260 条标注样本设计规则，测试前500+后500行。
输出：每个 content_type 的数量 + 标注数据准确率。
"""

import json, re, sys
from collections import Counter
import pandas as pd

# ==========================================================================
# 规则定义（基于标注数据分析）
# ==========================================================================

# 音游关键词（title或tags命中 → is_game=True）
GAME_KEYWORDS = [
    "PJSK", "プロセカ", "Project Sekai", "世界计划", "プロジェクトセカイ",
    "BangDream", "バンドリ", "BanG Dream", "少女乐团派对",
    "Phigros", "Arcaea", "Deemo", "Cytus", "Cytus II",
    "CHUNITHM", "チュウニズム", "中二节奏",
    "maimai", "舞萌", "osu!", "D4DJ",
    "Muse Dash", "VOEZ", "太鼓达人", "太鼓の達人",
    "音乐游戏", "音游",
]

# 翻唱/改编关键词（title或tags命中 → is_cover=True）
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

# VOCALOID/虚拟歌姬 标签（反向验证：tags中有这些 → 确实与虚拟歌姬相关）
VOCALOID_TAGS = [
    "VOCALOID", "vocaloid", "VOCALOID殿堂入り", "VOCALOID3",
    "CeVIO", "CeVIO AI", "UTAU", "ボカロ", "ボーカロイド",
    "V家", "術力口", "术力口", "虚拟歌姬", "虚拟歌手", "虚拟偶像",
]

# 歌姬名称 tag（tags中有这些 → 确实是音乐内容）
SINGER_TAGS = [
    "初音ミク", "初音未来", "初音MIKU", "Miku", "MIKU",
    "GUMI", "鏡音リン", "鏡音レン", "镜音RIN", "镜音LEN",
    "巡音ルカ", "巡音LUKA", "MEIKO", "KAITO", "MAYU",
    "結月ゆかり", "结月缘", "Fukase", "Flower", "v_flower",
    "可不", "重音テト", "重音TETO", "ONE", "OИE",
    "IA小天使", "IA ROCKS", "IA GLOWB",
    "洛天依", "言和", "乐正绫",
]

# "other"信号词（标题含这些 → 可能是 ia_related 而非 ia_music）
NON_SONG_KEYWORDS = [
    "人物志", "科普", "编年史", "发生了什么", "小课堂",
    "盘点", "周刊", "RANKING", "ランキング", "推荐",
    "纪录片", "官方频", "官方OFFICIAL", "OFFICIAL",
    "在家动手", "猜歌", "介绍", "小剧场", "同居",
    "实验", "靶场", "样本试听", "声音样本",
    "手书", "手描", "描改", "绘画", "MMD",
    "演唱会", "巡演", "现场", "超Party", "超PARTY",
    "来了", "新宝岛",
    "画师", "COS", "Cosplay",
    "剧场", "VOICEROID", "VOICEROID剧场",
    "语调教", "跨语种",
    "声库", "设计", "简介", "这到底是什么",
    "娇喘", "信念感",
    "组曲", "猜曲", "听力", "请问",
    "教程", "教学", "入门",
    "PK", "对决", "哪个", "比较",
    "P主推荐", "P主人物",
    "授权汉化",
    "宣布", "发表",
    "不幸", "自杀", "去世",
    "新闻", "速报",
    "报告", "统计", "评论", "WOTA艺"
]

# 无关信号词（title含这些且tags无虚拟歌姬 → irrelevant）
IRRELEVANT_SIGNALS = [
    "战争雷霆", "坦克", "攻击机", "战斗机", "军舰",
    "AI生成", "恐龙快打", "I want to eat",
    "Roblox", "roblox", "物品避难所", "item asylum",
    "装甲核心", "ACVI",
    "Apex", "APEX", "CSGO", "Valorant",
    "赛博朋克", "巫师3",
    "内鬼", "PVZ", "我的世界", "饥荒",
    "梗图", "meme", "搞笑配音",
    "荒野乱斗", "Brawl",
    "绝区零", "鸣潮",
    "假面骑士", "奥特曼",
]


# ==========================================================================
# 分类函数
# ==========================================================================

def classify(title, tags, category):
    """
    输入: title(str), tags(str), category(str)
    输出: {
        "content_type": "ia_music" | "ia_related" | "irrelevant",
        "is_game": bool,
        "is_cover": bool,
        "rule": str  # 命中的规则名（调试用）
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

    # ---- 分类判定 ----

    # 1. 强信号：tags 含虚拟歌姬标签 → ia_music 或 ia_related
    has_vocaloid_tag = any(vt.upper() in g_upper for vt in VOCALOID_TAGS)
    has_singer_tag = any(st.upper() in g_upper for st in SINGER_TAGS)
    has_ia_tag = any("IA" == t.upper() or "IA" in [x.upper() for x in t_upper.split(",")]
                     for t in tags_list)

    # 展开：检查 tags 中是否有"IA"标签（精确匹配或包含）
    has_ia_exact = False
    for t in tags_list:
        tu = t.upper()
        if tu == "IA" or tu.startswith("IA ") or "IA_" in tu or tu.endswith(" IA"):
            has_ia_exact = True
            break
    # 也检查 IA 作为子串在 tags 中出现（如 "IA小天使", "IA_-ARIA..."）
    has_ia_loose = has_ia_exact or any("IA" in t.upper() for t in tags_list)

    music_signal = has_vocaloid_tag or has_singer_tag or has_ia_exact
    any_ia = has_ia_loose or has_vocaloid_tag or has_singer_tag

    # 2. irrelevant 强信号：tags 中完全没有任何虚拟歌姬标记
    if not any_ia:
        # 补充检查 category
        cat_upper = (category or "").upper()
        music_cats = ["VOCALOID", "UTAU", "音MAD", "翻唱", "演奏", "音乐"]
        if not any(mc in cat_upper for mc in music_cats):
            return {"content_type": "irrelevant", "is_game": is_game, "is_cover": is_cover, "rule": "no_signal"}

    # 3. 标题含无关强信号 → irrelevant
    for kw in IRRELEVANT_SIGNALS:
        if kw.upper() in t_upper:
            return {"content_type": "irrelevant", "is_game": is_game, "is_cover": is_cover, "rule": f"irrelevant_kw:{kw}"}

    # 4. 标题含 non-song 关键词 → ia_related
    for kw in NON_SONG_KEYWORDS:
        if kw.upper() in t_upper:
            return {"content_type": "ia_related", "is_game": is_game, "is_cover": is_cover, "rule": f"nonsong:{kw}"}

    # 5. 有音乐信号 → ia_music
    if music_signal:
        return {"content_type": "ia_music", "is_game": is_game, "is_cover": is_cover, "rule": "music_signal"}

    # 6. 没有任何信号 → 根据 category 兜底
    cat_upper = (category or "").upper()
    if "VOCALOID" in cat_upper or "UTAU" in cat_upper or "翻唱" in cat_upper:
        return {"content_type": "ia_music", "is_game": is_game, "is_cover": is_cover, "rule": "category_vocaloid"}
    if "MMD" in cat_upper or "手书" in cat_upper:
        return {"content_type": "ia_related", "is_game": is_game, "is_cover": is_cover, "rule": "category_mmd"}
    if "游戏" in cat_upper or "日常" in cat_upper or "数码" in cat_upper:
        return {"content_type": "irrelevant", "is_game": is_game, "is_cover": is_cover, "rule": "category_gaming"}

    return {"content_type": "ia_related", "is_game": is_game, "is_cover": is_cover, "rule": "fallback"}


# ==========================================================================
# 测试标注样本
# ==========================================================================

def test_labeled_samples():
    """在 260 条标注上测试准确率"""
    with open("data/labeled_samples.json", encoding="utf-8") as f:
        samples = json.load(f)

    # 标注到三分类的映射
    LABEL_MAP = {
        "game_cover":        "ia_music",
        "vocaloid_original": "ia_music",
        "vocaloid_cover":    "ia_music",
        "other":             "ia_related",
        "irrelevant":        "irrelevant",
    }
    IS_GAME_TRUE  = {"game_cover"}
    IS_COVER_TRUE = {"vocaloid_cover"}

    correct = 0
    correct_game = 0
    correct_cover = 0
    total_game = 0
    total_cover = 0
    errors = Counter()

    for s in samples:
        result = classify(s["title"], s.get("tags", ""), s.get("category", ""))

        # 检查 content_type
        expected_type = LABEL_MAP[s["content_type"]]
        if result["content_type"] == expected_type:
            correct += 1
        else:
            errors[(s["content_type"], result["content_type"])] += 1

        # 检查 is_game
        expected_game = s["content_type"] in IS_GAME_TRUE
        if expected_game:
            total_game += 1
            if result["is_game"] == expected_game:
                correct_game += 1

        # 检查 is_cover
        expected_cover = s["content_type"] in IS_COVER_TRUE
        if expected_cover:
            total_cover += 1
            if result["is_cover"] == expected_cover:
                correct_cover += 1

    total = len(samples)
    print(f"\n{'='*60}")
    print(f"标注样本测试 ({total} 条)")
    print(f"{'='*60}")
    print(f"content_type 准确率: {correct}/{total} ({correct/total:.1%})")
    print(f"is_game 召回率:      {correct_game}/{total_game} ({correct_game/total_game:.1%})")
    print(f"is_cover 召回率:     {correct_cover}/{total_cover} ({correct_cover/total_cover:.1%})")

    if errors:
        print(f"\n错误分布 (top 10):")
        for (true_label, pred), count in errors.most_common(10):
            print(f"  {true_label} → {pred}: {count}")


# ==========================================================================
# 测试 CSV 数据
# ==========================================================================

def test_csv_data():
    """测试 ia_music_data.csv 的前500和后500行"""
    df = pd.read_csv("data/ia_music_data.csv")
    df = df.drop_duplicates(subset=["bvid", "page"])

    # 取前500 + 后500
    n = min(500, len(df))
    test_set = pd.concat([df.head(n), df.tail(n)]).drop_duplicates()
    print(f"\n{'='*60}")
    print(f"CSV 测试 (前{n}+后{n}, 去重后 {len(test_set)} 行)")
    print(f"{'='*60}")

    results = []
    for _, row in test_set.iterrows():
        result = classify(
            str(row.get("title", "")),
            str(row.get("tags", "")),
            str(row.get("category", "")),
        )
        results.append(result)

    # 统计
    type_counts = Counter(r["content_type"] for r in results)
    game_count = sum(1 for r in results if r["is_game"])
    cover_count = sum(1 for r in results if r["is_cover"])

    print(f"\ncontent_type 分布:")
    for label in ["ia_music", "ia_related", "irrelevant"]:
        count = type_counts.get(label, 0)
        pct = count / len(results) * 100
        print(f"  {label:15s}: {count:>5} ({pct:5.1f}%)")
    print(f"\n布尔标记:")
    print(f"  is_game:  {game_count} ({game_count/len(results)*100:.1f}%)")
    print(f"  is_cover: {cover_count} ({cover_count/len(results)*100:.1f}%)")

    # 命中规则分布
    rule_counts = Counter(r["rule"] for r in results)
    print(f"\n命中规则分布 (top 10):")
    for rule, count in rule_counts.most_common(10):
        print(f"  {rule}: {count}")

    # 每类抽查
    print(f"\n每类抽查 5 条:")
    for label in ["ia_music", "ia_related", "irrelevant"]:
        subset = [(results[i], test_set.iloc[i])
                  for i in range(len(results))
                  if results[i]["content_type"] == label]
        print(f"\n--- {label} ({len(subset)}条) ---")
        for r, row in subset[:5]:
            print(f"  [game={r['is_game']}, cover={r['is_cover']}, rule={r['rule']}]")
            print(f"    {row['title'][:70]}")

    # 导出分类结果为三个 CSV
    print(f"\n{'='*60}")
    print(f"导出分类结果 CSV")
    print(f"{'='*60}")

    for label in ["ia_music", "ia_related", "irrelevant"]:
        # 收集该类的行索引
        indices = [i for i in range(len(results)) if results[i]["content_type"] == label]
        rows = test_set.iloc[indices].copy()
        # 添加分类列
        rows["regex_content_type"] = label
        rows["regex_is_game"] = [results[i]["is_game"] for i in indices]
        rows["regex_is_cover"] = [results[i]["is_cover"] for i in indices]
        rows["regex_rule"] = [results[i]["rule"] for i in indices]
        # 精简输出列
        out_cols = ["bvid", "page", "title", "author", "tags", "category", "play_count",
                     "regex_content_type", "regex_is_game", "regex_is_cover", "regex_rule"]
        out_cols = [c for c in out_cols if c in rows.columns]
        fname = f"data/regex_{label}.csv"
        rows[out_cols].to_csv(fname, index=False, encoding="utf-8-sig")
        print(f"  {fname}: {len(rows)} 行")


# ==========================================================================
# 主流程
# ==========================================================================

if __name__ == "__main__":
    test_labeled_samples()
    test_csv_data()
