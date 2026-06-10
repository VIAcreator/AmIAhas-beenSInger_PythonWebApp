"""
标题解析原型测试程序 v2
改进：全角归一化 + P主启发式 + -IA后缀去除 + tags补全
"""

import re
import pandas as pd
from collections import Counter

# ==========================================================================
# 工具函数
# ==========================================================================

def normalize_fullwidth(text: str) -> str:
    """全角数字/英文字母/符号 → 半角。"""
    result = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:  # 全角标点+数字+字母
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:  # 全角空格
            result.append(' ')
        else:
            result.append(ch)
    return "".join(result)


# ==========================================================================
# 阶段0：已知 P主 列表
# ==========================================================================

_raw_creators = {
    "Orangestar", "orangestar", "OrangeStar",
    "kemu", "KEMU",
    "じん", "Jin", "自然の敵P", "自然之敌P", "じん（自然の敵P）", "じん(自然の敵P)",
    "梅とら", "まふまふ",
    "r-906", "Guiano", "guiano", 
    "傘村トータ", "*Luna", "ねじ式", "ATOLS", "うたたP",
    "ギガP", "ナナホシ管弦楽団", "花之祭P",
    "神無月P", "やいり", "すいっち", "VelecTi",
    "夏山よつぎ", "150P", "乙P", "ろくろ", "にほしか",
    "samfree", "out of survice", "Diarays", "PTL0★",
    "ちいたな", "メル", "PolyphonicBranch",
    "yksb", "ぐちり", "はなぽわんわんP", "Adeliae",
    "ねこぼーろ", "沙汰"
}
# 归一化后的已知P主集合
KNOWN_CREATORS = {normalize_fullwidth(c) for c in _raw_creators}

NOT_CREATOR_PATTERNS = [
    r'feat\.?\s*(IA|初音|MIKU)', r'【[^】]*】', r'^[\)\)D\]】]',
    r'初投稿', r'授权', r'搬运', r'オリジナル', r'カバー', r'Cover',
    r'部分', r'PV', r'MV', r'^$', r'^\d+$',
]


# ==========================================================================
# 阶段1：信息提取
# ==========================================================================

NOT_CREATOR_TOKENS = {
    "VOCALOID", "vocaloid", "Vocaloid", "VOCALOID3", "UTAU", "CeVIO", "CeVIO AI",
    "IA", "ia", "IA小天使", "IA ROCKS", "IA GLOWB", "イア", "伊爱",
    "初音ミク", "初音未来", "GUMI", "MEIKO", "KAITO", "MAYU",
    "鏡音リン", "鏡音レン", "巡音ルカ", "結月ゆかり",
    "ONE", "OИE", "Fukase", "v_flower", "可不", "重音テト", "Lily",
    "オリジナル", "オリジナル曲", "カバー", "Cover", "cover",
    "4K", "MV", "PV", "MMD", "手书", "本家投稿", "中文字幕", "授权转载",
    "VOCALOID殿堂入り", "VOCAROCK", "無色透名祭",
    "ゲキヤク", "STARDUST", "音街ウナ"  # IA ROCKS tracks
}

def extract_bracket_tokens(title: str) -> dict:
    """
    提取【】中内容并分类。加入全角归一化。
    """
    tokens = re.findall(r'【([^】]+)】', title)
    result = {"vocal_singers": [], "creators": [], "actions": [], "others": []}

    VOCAL_SINGERS = {"IA", "IA ROCKS", "IA :[R]", "IA:[R]", "初音ミク", "GUMI",
                     "鏡音リン", "鏡音レン", "巡音ルカ", "結月ゆかり", "ONE", "OИE",
                     "Fukase", "MAYU", "Lily", "VY1", "VY2", "可不", "重音テト",
                     "IA小天使", "IA GLOWB", "歌愛ユキ", "音街ウナ"}
    ACTIONS = {"オリジナル曲", "オリジナル", "原创曲", "搬运", "转载", "授权转载",
               "翻唱", "カバー", "cover", "Cover", "本家投稿", "中文字幕",
               "4K", "MV", "PV", "MMD", "手书", "IA翻唱", "IAオリジナル曲", "PV付", "日语翻唱"}

    for i, token in enumerate(tokens):
        t = normalize_fullwidth(token.strip())
        if t in VOCAL_SINGERS or any(vs in t for vs in ["IA", "初音", "GUMI", "鏡音", "巡音", "結月"]):
            if any(t.startswith(vs) for vs in VOCAL_SINGERS) or t in VOCAL_SINGERS:
                result["vocal_singers"].append(t)
            else:
                result["others"].append(t)
        elif t in KNOWN_CREATORS:
            result["creators"].append(t)
        elif t in ACTIONS:
            result["actions"].append(t)
        elif re.match(r'^[A-Za-z0-9_・☆★]+P$', t):
            # 启发式：以P结尾的名称 → P主
            result["creators"].append(t)
            KNOWN_CREATORS.add(t)
        elif i >= 1 and t not in NOT_CREATOR_TOKENS and "オリジナル" not in t:
            # 改进1：非首位的【】既非歌姬也非动作 → 可能是P主
            result["creators"].append(t)
            KNOWN_CREATORS.add(t)
        else:
            result["others"].append(t)

    return result


def extract_feat_creator(title: str) -> str | None:
    """提取 feat. 前后的创作者名。"""
    # 模式1: P主 feat. IA — P主在feat前面（含假名/汉字）
    m = re.search(r'([A-Za-z0-9_★☆・ぁ-ゟ一-鿿]+)\s+feat\.?\s*IA', title, re.I)
    if m:
        name = m.group(1).strip()
        if name.upper() not in ("IA", "FEAT") and len(name) >= 2:
            return name
    # 模式2: feat.XXX — P主在feat后面
    m = re.search(r'feat\.?\s*([A-Za-z0-9_★☆＊*・ぁ-ゟ一-鿿]+)', title, re.I)
    if m:
        name = m.group(1).strip()
        if name.upper() not in ("IA",) and len(name) >= 2:
            return name
    return None


def extract_prefix_creator(title: str) -> str | None:
    """提取 'P主名 - 歌名' 格式的前缀P主。"""
    m = re.match(r'^([A-Za-z0-9_★☆・]+)\s*[-–—]\s*', title)
    if m:
        name = m.group(1).strip()
        if name.upper() not in ("IA",) and len(name) >= 3:
            return name
    return None


def extract_slash_creator(title: str) -> str | None:
    """提取 / 后的可能P主名。"""
    parts = title.split("/")
    if len(parts) < 2:
        return None
    candidate = normalize_fullwidth(parts[-1].strip())
    candidate = re.sub(r'【[^】]*】', '', candidate).strip()
    if not candidate or len(candidate) > 30:
        return None
    for pat in NOT_CREATOR_PATTERNS:
        if re.search(pat, candidate, re.I):
            return None
    if candidate.upper() in ("IA", "IA ", "初音ミク", "MIKU", "GUMI"):
        return None
    if candidate in KNOWN_CREATORS:
        return candidate
    if re.match(r'^[぀-ゟ゠-ヿ一-鿿A-Za-z0-9_・☆★]+$', candidate):
        return candidate
    return None


# ==========================================================================
# 阶段2：歌名提取
# ==========================================================================

def extract_song_name(title: str, creators: list, vocal_singers: list) -> str:
    """去除标签、P主名、-IA后缀和噪声，提取歌名。"""
    s = normalize_fullwidth(title)
    s = s[:2000]  # 实际标题最长约 200 字符

    # 0. 解码 HTML 实体 (&amp; → &, &#x27; → ', etc.)
    import html
    s = html.unescape(s)

    # 1. 去除 Niconico 视频 ID
    s = re.sub(r'[_\s]?(?:sm|so|nm)\d{4,}\b', '', s, flags=re.I).strip()
    # 2. 去除所有括号标签: 【】〖〗[]［］
    s = re.sub(r'【[^】]*】', '', s).strip()
    s = re.sub(r'〖[^〗]*〗', '', s).strip()
    s = re.sub(r'\[[^\]]*\]', '', s).strip()
    s = re.sub(r'［[^］]*］', '', s).strip()

    # 3. 去除 feat. 模式（含括号变体）
    s = re.sub(r'\(?\s*feat\.?\s*[^)]*\)?\s*', '', s, flags=re.I).strip()

    # 4. 去除 P主前缀: "P主名 - 歌名" → 歌名
    for creator in creators:
        s = re.sub(r'^' + re.escape(creator) + r'\s*[-–—]\s*', '', s).strip()

    # 5. 去除 / P主名后缀 + /歌手名 后缀
    for creator in creators:
        s = re.sub(r'\s*/\s*' + re.escape(creator) + r'\s*$', '', s).strip()
    # 通用 [/／]末尾剥离: 最后一段全是歌手名时去掉
    _singers = '|'.join(re.escape(k) for k in sorted(SINGER_KEYWORDS.keys(), key=len, reverse=True))
    s = re.sub(r'\s*[/／]\s*(?:' + _singers +
               r')(?:\s*[,&＆・／/x×\s]\s*(?:' + _singers + r'))*\s*$',
               '', s, flags=re.I).strip()

    # 6. 去除 -IA 后缀
    s = re.sub(r'\s*[-–/]\s*IA(\s*[;+]\s*\S+)?(\s*(version|ver\.?))?\s*$', '', s, flags=re.I).strip()
    s = re.sub(r'\s*/\s*IA\s*$', '', s, flags=re.I).strip()

    # 7. 去除末尾括号注释（英文/日文翻译）
    s = re.sub(r'\s*\([^)]{10,}\)\s*$', '', s).strip()
    s = re.sub(r'\s*（[^）]{10,}）\s*$', '', s).strip()

    # 8. 去除平台/标签噪声短语（无论位置）
    noise_phrases = [
        r'VOCALOID\s*オリジナル曲', r'VOCALOID\s*,\s*オリジナル曲',
        r'IA\s*オリジナル曲', r'IAオリジナル曲', r'オリジナル曲',
        r'ニコニコ動画', r'ニコニコ', r'niconico', 
    ]
    for pat in noise_phrases:
        s = re.sub(pat, '', s, flags=re.I).strip()

    # 9. 去除常见噪声后缀
    noise_suffixes = [
        r'\s*「[^」]*」\s*$',
        r'\s*[-–\s]\s*(Music Video|Official Video|MV|PV|Short Ver\.?|Full Ver\.?|Official MV)\s*$',
    ]
    for pat in noise_suffixes:
        s = re.sub(pat, '', s, flags=re.I).strip()

    # 10. 去除 x/× 连接的歌手名块
    s = re.sub(r'(?:' + _singers + r')\s*[x×]\s*(?:' + _singers + r'\s*[x×]\s*)*(?:' + _singers + r')\s*',
               '', s, flags=re.I).strip()
    # 清理残留的 x/× 分隔符
    s = re.sub(r'\s*[x×]\s*', '', s).strip()

    # 10. 清理 / 后的翻译片段
    s = re.sub(r'\s*/\s*[一-鿿][一-鿿\s]{2,}\s*$', '', s).strip()
    s = re.sub(r'\s*/\s*[぀-ヿー\s]{3,}\s*$', '', s).strip()
    # 通用: 末尾 /name 段 → 剥离（但不剥离纯中日文字符以防误杀歌名）
    s = re.sub(r'\s*/\s*[A-Za-z0-9_★☆・]{2,25}\s*$', '', s).strip()

    # 11. 清理多余空格、符号、+号连接
    s = re.sub(r'\s+\+\s+.+$', '', s).strip()
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.strip('/-｜|　 ')

    return s if s and len(s) > 1 else title


# ==========================================================================
# 阶段3：歌手检测
# ==========================================================================

# 歌手关键词 → 规范名称
SINGER_KEYWORDS = {
    "IA": "IA", "IA:[R]": "IA:[R]", "IA :[R]": "IA:[R]", "IA ROCKS": "IA ROCKS", "IA GLOWB": "IA GLOWB",
    "初音ミク": "初音ミク", "初音未来": "初音ミク", "初音MIKU": "初音ミク", "初音": "初音ミク", "Miku": "初音ミク",
    "鏡音リン": "鏡音リン", "鏡音レン": "鏡音レン", "镜音RIN": "鏡音リン", "镜音LEN": "鏡音レン",
    "巡音ルカ": "巡音ルカ", "巡音LUKA": "巡音ルカ",
    "MEIKO": "MEIKO", "KAITO": "KAITO",
    "GUMI": "GUMI", "Lily": "Lily", "MAYU": "MAYU",
    "結月ゆかり": "結月ゆかり", "结月缘": "結月ゆかり",
    "ONE": "ONE", "OИE": "ONE",
    "可不": "可不", "重音テト": "重音テト", "重音TETO": "重音テト",
    "v_flower": "Flower", "Flower": "Flower",
    "VY1": "VY1", "VY2": "VY2", "歌愛ユキ": "歌愛ユキ",
    "洛天依": "洛天依", "言和": "言和", "乐正绫": "乐正绫",
    "miki": "miki", "SF-A2 開発コード miki": "miki",
    "Prima": "Prima", "東北ずん子": "東北ずん子",
    "音街ウナ": "音街ウナ", "音街鳗": "音街ウナ",
}


def detect_singers(title: str, tags: str = "") -> str:
    """
    检测歌手：优先从标题匹配，标题无结果时从 tags 补充。

    输入: title, tags
    输出: "IA, 初音ミク, GUMI"（排序、去重、逗号分隔）
    """
    sorted_keys = sorted(SINGER_KEYWORDS.keys(), key=len, reverse=True)

    def _find_in(text: str) -> set:
        found = set()
        matched_positions = set()
        for kw in sorted_keys:
            pattern = re.escape(kw)
            for m in re.finditer(pattern, text, re.I):
                start, end = m.start(), m.end()
                if not any(start < p[1] and end > p[0] for p in matched_positions):
                    found.add(SINGER_KEYWORDS[kw])
                    matched_positions.add((start, end))
        return found

    # 优先标题（额外处理：x/× 连接的歌手如 "初音x巡音xGUMIxIA"）
    title_norm = normalize_fullwidth(title)
    # 拆分 x/×/＆/& 连接符 → 展开为独立歌手名方便匹配
    title_expanded = re.sub(r'[x×＆&]', ' ', title_norm)
    found = _find_in(title_expanded)

    # 标题无结果 → tags 兜底
    if not found:
        found = _find_in(normalize_fullwidth(tags or ""))

    if not found:
        found.add("IA")

    return ", ".join(sorted(found))


# ==========================================================================
# 阶段4：P主编排
# ==========================================================================

def assign_creator(title: str, tags: str, bracket_info: dict,
                   feat_creator: str | None, slash_creator: str | None,
                   prefix_creator: str | None) -> str:
    """按优先级确定 original_creator。"""
    # 1. 【】中的P主
    if bracket_info["creators"]:
        return bracket_info["creators"][0]

    # 2. / 后验证过的名称
    if slash_creator:
        return slash_creator

    # 3. feat. 模式
    if feat_creator:
        return feat_creator

    # 4. 前缀 P主 (P主 - 歌名)
    if prefix_creator:
        return prefix_creator

    # 5. tags 中匹配已知P主列表
    if tags:
        for tag in tags.split(","):
            tag = normalize_fullwidth(tag.strip())
            if tag in KNOWN_CREATORS:
                return tag

    return ""


# ==========================================================================
# 关键词映射库（手动维护，最长匹配优先）
# ==========================================================================

_keyword_map = None  # 延迟加载

def _load_keyword_map(path: str = "data/song_keywords.csv") -> pd.DataFrame:
    """加载关键词映射表，按关键词长度降序排序。"""
    global _keyword_map
    if _keyword_map is None:
        import os
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["_kw_len"] = df["keyword"].str.len()
            _keyword_map = df.sort_values("_kw_len", ascending=False)
    return _keyword_map


def match_keyword(title: str) -> dict | None:
    """
    查询映射库。多个 keyword 命中时取最长者。

    输入: title
    输出: {"song_name": str, "original_creator": str} 或 None
    """
    mapping = _load_keyword_map()
    if mapping is None or len(mapping) == 0:
        return None

    t_upper = normalize_fullwidth(title).upper()
    best = None

    for _, row in mapping.iterrows():
        kw = str(row["keyword"])
        # 已找到更长的匹配 → 跳过更短的
        if best and len(kw) <= len(best["keyword"]):
            continue
        if kw.upper() in t_upper:
            best = {"song_name": row["song_name"], "original_creator": row["original_creator"],
                    "keyword": kw}

    if best:
        return {"song_name": best["song_name"], "original_creator": best["original_creator"]}
    return None


# ==========================================================================
# 主解析函数
# ==========================================================================

def parse_title(title: str, tags: str = "") -> dict:
    if not title:
        return {"song_name": "", "original_creator": "", "vocal_singer": "", "is_reupload": False}

    bracket_info = extract_bracket_tokens(title)
    feat_creator = extract_feat_creator(title)
    slash_creator = extract_slash_creator(title)
    prefix_creator = extract_prefix_creator(title)

    all_creators = list(set(
        bracket_info["creators"] +
        ([slash_creator] if slash_creator else []) +
        ([feat_creator] if feat_creator else []) +
        ([prefix_creator] if prefix_creator else [])
    ))
    song_name = extract_song_name(title, all_creators, bracket_info["vocal_singers"])

    original_creator = assign_creator(title, tags, bracket_info, feat_creator, slash_creator, prefix_creator)

    vocal_singer = detect_singers(title, tags)

    # 关键词映射库覆写（最长匹配优先）
    kw_match = match_keyword(title)
    if kw_match:
        song_name = kw_match["song_name"]
        # 映射库中有明确P主（非空、非?）时覆盖正则结果
        kw_creator = str(kw_match["original_creator"]).strip()
        if kw_creator and kw_creator != "?":
            original_creator = kw_creator

    is_reupload = any(
        kw in " ".join(bracket_info["actions"] + bracket_info["others"])
        for kw in ["搬运", "转载", "授权转载"]
    )

    return {
        "song_name": song_name,
        "original_creator": original_creator,
        "vocal_singer": vocal_singer,
        "is_reupload": is_reupload,
    }


# ==========================================================================
# 测试 + 导出
# ==========================================================================

def main():
    # 使用标注好的 ia_music 数据集
    import csv
    samples = []
    with open("data/labeled_samples/music.csv", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        ti, gi = header.index("title"), header.index("tags")
        for row in reader:
            if len(row) > max(ti, gi):
                samples.append({
                    "title": row[ti],
                    "tags": row[gi],
                    "bvid": row[0] if len(row) > 0 else "",
                    "page": row[1] if len(row) > 1 else 1,
                })

    test = pd.DataFrame(samples)
    print(f"测试 {len(test)} 行 (来自 music.csv 标注集)\n")

    results = []
    for _, row in test.iterrows():
        r = parse_title(str(row["title"]), str(row.get("tags", "")))
        results.append(r)

    has_creator = sum(1 for r in results if r["original_creator"])
    print(f"P主 提取率: {has_creator}/{len(results)} ({has_creator/len(results):.0%})")

    creators = Counter(r["original_creator"] for r in results if r["original_creator"])
    print(f"Top P主: {', '.join(f'{k}({v})' for k,v in creators.most_common(8))}")

    # 抽查
    print(f"\n{'='*60}")
    print("抽查 10 条")
    print(f"{'='*60}")
    for i in [0, 10, 20, 40, 60, 80, 100, 120, 140, 180]:
        if i >= len(results): break
        row = test.iloc[i]
        r = results[i]
        creator_info = r['original_creator'] or '(无)'
        print(f"  [{creator_info}] {r['song_name'][:50]}")
        if r["is_reupload"]: print(f"    搬运")

    # 导出 CSV 供人工审核
    out_rows = []
    for i, r in enumerate(results):
        row = test.iloc[i]
        out_rows.append({
            "bvid": row.get("bvid", ""),
            "page": row.get("page", 1),
            "title": row["title"],
            "tags": row.get("tags", ""),
            "song_name": r["song_name"],
            "original_creator": r["original_creator"],
            "vocal_singer": r["vocal_singer"],
            "is_reupload": r["is_reupload"],
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv("data/title_parse_test.csv", index=False, encoding="utf-8-sig")
    print(f"\n导出: data/title_parse_test.csv ({len(out_df)} 行)")


if __name__ == "__main__":
    main()
