"""
正则分类测试程序 (v2 三分类标准)
=================================
基于 260 条标注样本设计规则，测试前500+后500行。
输出：每个 content_type 的数量 + 标注数据准确率。
"""

import re, os
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
    "洛天依", "言和", "乐正绫", "IA:[R]", "miki"
    ]

# "other"信号词（标题含这些 → 可能是 ia_related 而非 ia_music）
# 但若标题同时有"音乐强信号"（【IA】歌名格式、feat.IA等），则优先判为 ia_music
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
    # 排行榜/精选/纪录片类
    "排行", "TOP100", "TOP50", "BEST", "传送", "精选",
    "超过", "为止",
    "贺岁", "拜年", "单品", "原创动画", "人气",
    "传说曲", "殿堂", "神话曲", "名曲",
    "调查", "问卷", "投票", "第一期", "第二期", "第三期",
    "试着跳", "试着做了", "试着演奏", "试着画",
]


# 最高优先级无关词：标题或tags命中 → 直接 irrelevant，无视一切
HIGH_PRIORITY_IRRELEVANT = [
    "Roblox", "roblox", "物品避难所", "物品庇护所",
    "Item Asylum", "item asylum", "itemasylum", "ItemAsylum",
    "千问", "三角洲", "枪花", "漫剧",
]

# 普通无关信号词（title含这些 → irrelevant，但有 feat.IA 时豁免）
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


# ==========================================================================
# 分类函数
# ==========================================================================

def classify(title, tags, category):
    """
    输入: title(str), tags(str), category(str)
    输出: {
        "content_type": "ia_music" | "ia_related" | "irrelevant" | "suspicious",
        "is_game": bool,
        "is_cover": bool,
        "rule": str,
        "suspicious": bool,
        "suspicious_reason": str  # 可疑原因（suspicious=True 时有值）
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
    # tags 只有 IA 但没有 VOCALOID/歌姬 → IA 可能是 AI 的笔误或产品型号
    only_ia_signal = has_ia_exact and not has_vocaloid_tag and not has_singer_tag

    strong_title_music = bool(
        re.search(r'feat\.?\s*IA', title, re.I) or
        re.search(r'【IA[^】]*】', title) or
        re.search(r'[/／][^/／]+feat', title, re.I)
    )

    # 标题没有【IA】/feat.IA 格式 → 可能不是单曲投稿
    no_ia_format = not strong_title_music

    # 标题奇怪：很短、纯问句、含特殊字符
    clean_title = re.sub(r'【[^】]*】', '', title).strip()
    clean_title = re.sub(r'\[[^\]]*\]', '', clean_title).strip()
    weird_title = (
        len(clean_title) < 8 or
        "?" in title or "？" in title or
        "..." in title or "!!!" in title
    )

    # ---- 分类判定 ----

    result = {"is_game": is_game, "is_cover": is_cover, "rule": "", "suspicious": False, "suspicious_reason": ""}

    # 1. 最高优先级：命中强无关词 → 直接 irrelevant（无视音乐信号、feat.IA等一切）
    for kw in HIGH_PRIORITY_IRRELEVANT:
        if kw.upper() in t_upper or kw.upper() in g_upper:
            result.update({"content_type": "irrelevant", "rule": f"high_priority:{kw}"})
            return result

    # 2. irrelevant 强信号：tags 中完全没有任何虚拟歌姬标记，且标题无 feat.IA
    if not any_ia and not strong_title_music:
        cat_upper = (category or "").upper()
        music_cats = ["VOCALOID", "UTAU", "音MAD", "翻唱", "演奏", "音乐"]
        if not any(mc in cat_upper for mc in music_cats):
            result.update({"content_type": "irrelevant", "rule": "no_signal"})
            return result

    # 3. 标题含无关强信号 → irrelevant
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
        # 有 non-song 关键词，但有音乐信号 → 混合信号，标记可疑
        if music_signal:
            result.update({
                "content_type": "ia_related", "rule": f"nonsong_mixed:{hit_nonsong}",
                "suspicious": True,
                "suspicious_reason": f"标题含'{hit_nonsong}'(非歌曲信号)但tags有音乐标签，需LLM确认"
            })
            return result
        else:
            result.update({"content_type": "ia_related", "rule": f"nonsong:{hit_nonsong}"})
            return result

    # 5. 有音乐信号但标题没有【IA】格式 → 可能是盘点/周边，标记可疑
    if music_signal and no_ia_format:
        if weird_title:
            result.update({
                "content_type": "ia_music", "rule": "music_weird_title",
                "suspicious": True,
                "suspicious_reason": f"tags有IA/VOCALOID但标题格式非标准歌曲(clean='{clean_title[:30]}')，需LLM确认"
            })
            return result
        # tags有音乐信号，标题正常 → 高置信度
        result.update({"content_type": "ia_music", "rule": "music_signal"})
        return result

    # 6. 有音乐信号 + 标题格式明确 → 高置信度 ia_music
    if music_signal or strong_title_music:
        result.update({"content_type": "ia_music", "rule": "music_signal"})
        return result

    # 7. 没有任何信号 → 标记可疑，交给 LLM
    cat_upper = (category or "").upper()
    if "VOCALOID" in cat_upper or "UTAU" in cat_upper or "翻唱" in cat_upper:
        result.update({
            "content_type": "ia_music", "rule": "category_vocaloid",
            "suspicious": True,
            "suspicious_reason": f"无明确信号，仅靠分区'{category}'推断为ia_music"
        })
        return result
    if "MMD" in cat_upper or "手书" in cat_upper:
        result.update({
            "content_type": "ia_related", "rule": "category_mmd",
            "suspicious": True,
            "suspicious_reason": f"无明确信号，仅靠分区'{category}'推断"
        })
        return result
    if "游戏" in cat_upper or "日常" in cat_upper or "数码" in cat_upper:
        result.update({
            "content_type": "irrelevant", "rule": "category_gaming",
            "suspicious": True,
            "suspicious_reason": f"无明确信号，仅靠分区'{category}'推断"
        })
        return result

    result.update({
        "content_type": "ia_related", "rule": "fallback",
        "suspicious": True,
        "suspicious_reason": "无任何明确信号，默认归为ia_related"
    })
    return result


# ==========================================================================
# 测试标注样本
# ==========================================================================

def test_labeled_samples():
    """在 data/labeled_samples/*.csv 上测试准确率"""
    import glob

    # 新标注数据：文件名即标签
    FILE_LABELS = {
        "music":       "ia_music",
        "irrelevent":  "irrelevant",
        "related":     "ia_related",
    }

    samples = []
    for fname, label in FILE_LABELS.items():
        path = f"data/labeled_samples/{fname}.csv"
        if not os.path.exists(path):
            print(f"  跳过: {path} (不存在)")
            continue
        df = pd.read_csv(path, on_bad_lines='skip', encoding='utf-8')
        for _, row in df.iterrows():
            samples.append({
                "title":    str(row.get("title", "")),
                "tags":     str(row.get("tags", "")),
                "category": str(row.get("category", "")),
                "expected": label,
            })

    if not samples:
        print("  无标注数据，跳过测试")
        return

    correct = 0
    suspicious_correct = 0
    suspicious_total = 0
    errors = Counter()

    for s in samples:
        result = classify(s["title"], s["tags"], s["category"])
        if result["content_type"] == s["expected"]:
            correct += 1
        else:
            errors[(s["expected"], result["content_type"])] += 1
        if result["suspicious"]:
            suspicious_total += 1
            if result["content_type"] == s["expected"]:
                suspicious_correct += 1

    total = len(samples)
    print(f"\n{'='*60}")
    print(f"新标注样本测试 ({total} 条)")
    print(f"{'='*60}")
    print(f"content_type 准确率: {correct}/{total} ({correct/total:.1%})")
    if suspicious_total > 0:
        print(f"  其中可疑({suspicious_total}条)的准确率: {suspicious_correct}/{suspicious_total} ({suspicious_correct/suspicious_total:.1%})")

    # 每类准确率
    print(f"\n各类准确率:")
    for label in ["ia_music", "ia_related", "irrelevant"]:
        items = [s for s in samples if s["expected"] == label]
        ok = sum(1 for s in items if classify(s["title"], s["tags"], s["category"])["content_type"] == label)
        pct = f"{ok/len(items):.0%}" if items else "N/A"
        print(f"  {label:15s}: {ok}/{len(items)} ({pct})")

    if errors:
        print(f"\n错误分布 (top 10):")
        for (true_label, pred), count in errors.most_common(10):
            print(f"  {true_label} → {pred}: {count}")

    # 输出所有误判条目
    print(f"\n{'='*60}")
    print(f"所有误判条目")
    print(f"{'='*60}")

    wrong_rows = []
    for s in samples:
        result = classify(s["title"], s["tags"], s["category"])
        if result["content_type"] != s["expected"]:
            print(f"  [{s['expected']} → {result['content_type']}, rule={result['rule']}]")
            print(f"    {s['title'][:80]}")
            print(f"    tags: {s['tags'][:80]}")
            print()
            wrong_rows.append({
                "title": s["title"], "tags": s["tags"], "category": s["category"],
                "expected": s["expected"], "predicted": result["content_type"],
                "rule": result["rule"], "is_game": result["is_game"],
                "is_cover": result["is_cover"], "suspicious": result["suspicious"],
            })

    # 导出误判到分类 CSV
    if wrong_rows:
        import os as _os
        _os.makedirs("data", exist_ok=True)
        for label, fname in [("ia_music", "music_wrong"), ("ia_related", "related_wrong"), ("irrelevant", "irrelevant_wrong")]:
            subset = [r for r in wrong_rows if r["expected"] == label]
            if subset:
                pd.DataFrame(subset).to_csv(f"data/{fname}.csv", index=False, encoding="utf-8-sig")
                print(f"  导出: data/{fname}.csv ({len(subset)} 条)")


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
    suspicious_count = sum(1 for r in results if r["suspicious"])
    print(f"\n布尔标记:")
    print(f"  is_game:     {game_count} ({game_count/len(results)*100:.1f}%)")
    print(f"  is_cover:    {cover_count} ({cover_count/len(results)*100:.1f}%)")
    print(f"  suspicious:  {suspicious_count} ({suspicious_count/len(results)*100:.1f}%) ← 需LLM确认")

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
        indices = [i for i in range(len(results)) if results[i]["content_type"] == label]
        rows = test_set.iloc[indices].copy()
        rows["regex_content_type"] = label
        rows["regex_is_game"] = [results[i]["is_game"] for i in indices]
        rows["regex_is_cover"] = [results[i]["is_cover"] for i in indices]
        rows["regex_rule"] = [results[i]["rule"] for i in indices]
        rows["suspicious"] = [results[i]["suspicious"] for i in indices]
        rows["suspicious_reason"] = [results[i]["suspicious_reason"] for i in indices]
        out_cols = ["bvid", "page", "title", "author", "tags", "category", "play_count",
                     "regex_content_type", "regex_is_game", "regex_is_cover",
                     "regex_rule", "suspicious", "suspicious_reason"]
        out_cols = [c for c in out_cols if c in rows.columns]
        fname = f"data/regex_{label}.csv"
        rows[out_cols].to_csv(fname, index=False, encoding="utf-8-sig")
        print(f"  {fname}: {len(rows)} 行")

    # 4. 导出可疑视频（混合信号 / 正则不确定）
    suspicious_idx = [i for i in range(len(results)) if results[i]["suspicious"]]
    if suspicious_idx:
        sus_rows = test_set.iloc[suspicious_idx].copy()
        sus_rows["regex_content_type"] = [results[i]["content_type"] for i in suspicious_idx]
        sus_rows["regex_is_game"] = [results[i]["is_game"] for i in suspicious_idx]
        sus_rows["regex_is_cover"] = [results[i]["is_cover"] for i in suspicious_idx]
        sus_rows["regex_rule"] = [results[i]["rule"] for i in suspicious_idx]
        sus_rows["suspicious_reason"] = [results[i]["suspicious_reason"] for i in suspicious_idx]
        out_cols = ["bvid", "page", "title", "author", "tags", "category", "play_count",
                     "regex_content_type", "regex_is_game", "regex_is_cover",
                     "regex_rule", "suspicious_reason"]
        out_cols = [c for c in out_cols if c in sus_rows.columns]
        sus_rows[out_cols].to_csv("data/regex_suspicious.csv", index=False, encoding="utf-8-sig")
        print(f"  data/regex_suspicious.csv: {len(sus_rows)} 行（需LLM确认）")


# ==========================================================================
# 主流程
# ==========================================================================

if __name__ == "__main__":
    test_labeled_samples()
    test_csv_data()
