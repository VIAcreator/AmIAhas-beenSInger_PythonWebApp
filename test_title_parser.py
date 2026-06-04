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
    "ねこぼーろ", "沙汰",
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

def extract_bracket_tokens(title: str) -> dict:
    """
    提取【】中内容并分类。加入全角归一化。
    """
    tokens = re.findall(r'【([^】]+)】', title)
    result = {"vocal_singers": [], "creators": [], "actions": [], "others": []}

    VOCAL_SINGERS = {"IA", "IA ROCKS", "IA :[R]", "IA:[R]", "初音ミク", "GUMI",
                     "鏡音リン", "鏡音レン", "巡音ルカ", "結月ゆかり", "ONE", "OИE",
                     "Fukase", "MAYU", "Lily", "VY1", "VY2", "可不", "重音テト",
                     "IA小天使", "IA GLOWB", "歌愛ユキ"}
    ACTIONS = {"オリジナル曲", "オリジナル", "原创曲", "搬运", "转载", "授权转载",
               "翻唱", "カバー", "cover", "Cover", "本家投稿", "中文字幕",
               "4K", "MV", "PV", "MMD", "手书", "IA翻唱", "IAオリジナル曲"}

    for token in tokens:
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
        else:
            result["others"].append(t)

    return result


def extract_feat_creator(title: str) -> str | None:
    """提取 feat.XXX 中的创作者名。"""
    m = re.search(r'feat\.?\s*([A-Za-z0-9_★☆＊*・]+)', title, re.I)
    if m:
        name = m.group(1).strip()
        if name.upper() not in ("IA",):
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

    # 1. 去除所有【】标签
    s = re.sub(r'【[^】]*】', '', s).strip()

    # 2. 去除 feat.XXX 模式
    s = re.sub(r'feat\.?\s*\S+', '', s, flags=re.I).strip()

    # 3. 去除 / P主名 后缀
    for creator in creators:
        s = re.sub(r'\s*/\s*' + re.escape(creator) + r'\s*$', '', s).strip()

    # 4. 去除 -IA / -IA【PV】 等歌姬后缀
    s = re.sub(r'\s*[-–]\s*IA\s*(【[^】]*】)?\s*$', '', s).strip()
    # 也去掉末尾孤立的 - IA
    s = re.sub(r'\s*[-–]\s*IA\s*$', '', s).strip()

    # 5. 去除 "(...)" 中的非歌名内容（如英文翻译），但保留歌名关键部分
    # 只在前30字符后出现的括号内容视为注释
    if len(s) > 40:
        s = re.sub(r'\s*\([^)]{15,}\)\s*$', '', s).strip()

    # 6. 去除常见噪声后缀
    noise_suffixes = [
        r'\s*「[^」]*」\s*$',
        r'\s*\[[^\]]*\]\s*$',
        r'\s*（[^）]*）\s*$',
    ]
    for pat in noise_suffixes:
        s = re.sub(pat, '', s).strip()

    # 7. 清理
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.strip('/-｜|　 ')

    return s if s and len(s) > 1 else title


# ==========================================================================
# 阶段3：P主编排
# ==========================================================================

def assign_creator(title: str, tags: str, bracket_info: dict,
                   feat_creator: str | None, slash_creator: str | None) -> str:
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

    # 4. tags 中匹配已知P主列表
    if tags:
        for tag in tags.split(","):
            tag = normalize_fullwidth(tag.strip())
            if tag in KNOWN_CREATORS:
                return tag

    return ""


# ==========================================================================
# 主解析函数
# ==========================================================================

def parse_title(title: str, tags: str = "") -> dict:
    if not title:
        return {"song_name": "", "original_creator": "", "vocal_singer": "", "is_reupload": False}

    bracket_info = extract_bracket_tokens(title)
    feat_creator = extract_feat_creator(title)
    slash_creator = extract_slash_creator(title)

    all_creators = bracket_info["creators"] + ([slash_creator] if slash_creator else [])
    song_name = extract_song_name(title, all_creators, bracket_info["vocal_singers"])

    original_creator = assign_creator(title, tags, bracket_info, feat_creator, slash_creator)

    vocal_singer = bracket_info["vocal_singers"][0] if bracket_info["vocal_singers"] else "IA"

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
