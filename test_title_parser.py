"""
标题解析原型测试程序
四阶段：信息提取 → 歌名提取 → P主编排 → 翻唱补全
测试：前200行，输出解析结果供人工审查
"""

import re
import pandas as pd
from collections import Counter

# ==========================================================================
# 阶段0：已知 P主 列表（从 tags 统计中提取的高频 P主名）
# ==========================================================================

KNOWN_CREATORS = {
    "Orangestar", "orangestar", "OrangeStar",
    "kemu", "KEMU",
    "じん", "Jin", "自然の敵P", "じん（自然の敵P）", "自然之敌P",
    "梅とら", "まふまふ",
    "r-906", "Guiano", "guiano",
    "傘村トータ", "*Luna", "ねじ式", "ATOLS", "うたたP",
    "ギガP", "ナナホシ管弦楽団", "花之祭P",
    "神無月P", "やいり", "すいっち", "VelecTi",
    "夏山よつぎ", "150P", "乙P", "ろくろ", "にほしか",
    "samfree", "out of survice", "Diarays", "PTL0★",
    "ちいたな", "メル", "PolyphonicBranch",
    "yksb", "ぐちり", "はなぽわんわんP", "Adeliae",
    "ねこぼーろ", "沙汰", "メル",
}

# 常见非P主词（/ 后出现，需要排除）
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
    提取【】中内容并分类。

    返回:
        {"vocal_singers": ["IA"],
         "creators": ["kemu"],
         "actions": ["オリジナル曲", "搬运"],
         "others": ["VOCALOID"]}
    """
    tokens = re.findall(r'【([^】]+)】', title)
    result = {"vocal_singers": [], "creators": [], "actions": [], "others": []}

    VOCAL_SINGERS = {"IA", "IA ROCKS", "IA :[R]", "IA:[R]", "初音ミク", "GUMI",
                     "鏡音リン", "鏡音レン", "巡音ルカ", "結月ゆかり", "ONE", "OИE",
                     "Fukase", "MAYU", "Lily", "VY1", "VY2", "可不", "重音テト",
                     "IA小天使", "IA GLOWB"}
    ACTIONS = {"オリジナル曲", "オリジナル", "原创曲", "搬运", "转载", "授权转载",
               "翻唱", "カバー", "cover", "Cover", "本家投稿", "中文字幕",
               "4K", "MV", "PV", "MMD", "手书"}

    for token in tokens:
        t = token.strip()
        if t in VOCAL_SINGERS or any(vs in t for vs in ["IA", "初音", "GUMI", "鏡音", "巡音", "結月"]):
            # 歌姬相关
            if any(t.startswith(vs) for vs in VOCAL_SINGERS) or t in VOCAL_SINGERS:
                result["vocal_singers"].append(t)
            else:
                result["others"].append(t)
        elif t in KNOWN_CREATORS:
            result["creators"].append(t)
        elif t in ACTIONS:
            result["actions"].append(t)
        else:
            result["others"].append(t)

    return result


def extract_feat_creator(title: str) -> str | None:
    """提取 feat.XXX / feat XXX 中的创作者名。"""
    m = re.search(r'feat\.?\s*([A-Za-z0-9_★☆＊*・]+)', title, re.I)
    if m:
        name = m.group(1).strip()
        if name.upper() not in ("IA",):
            return name
    return None


def extract_slash_creator(title: str) -> str | None:
    """提取 / 后的可能P主名，排除已知噪声。"""
    parts = title.split("/")
    if len(parts) < 2:
        return None

    candidate = parts[-1].strip()
    # 去除【】标签
    candidate = re.sub(r'【[^】]*】', '', candidate).strip()
    if not candidate or len(candidate) > 30:
        return None

    # 排除噪声
    for pat in NOT_CREATOR_PATTERNS:
        if re.search(pat, candidate, re.I):
            return None

    if candidate in KNOWN_CREATORS:
        return candidate
    # 日语P主名（含假名/汉字）
    if re.match(r'^[぀-ゟ゠-ヿ一-鿿A-Za-z0-9_・☆★]+$', candidate):
        return candidate
    return None


# ==========================================================================
# 阶段2：歌名提取
# ==========================================================================

def extract_song_name(title: str, creators: list, vocal_singers: list) -> str:
    """去除标签、P主名和噪声后缀，提取歌名。"""
    s = title

    # 1. 去除所有【】标签
    s = re.sub(r'【[^】]*】', '', s).strip()

    # 2. 去除 feat.XXX 模式
    s = re.sub(r'feat\.?\s*\S+', '', s, flags=re.I).strip()

    # 3. 去除 / P主名 后缀
    for creator in creators:
        s = re.sub(r'\s*/\s*' + re.escape(creator) + r'\s*$', '', s).strip()

    # 4. 去除常见噪声后缀
    noise_suffixes = [
        r'\s*「[^」]*」\s*$',   # 「授权转载」
        r'\s*\[[^\]]*\]\s*$',   # [授权转载]
        r'\s*（[^）]*）\s*$',   # （授权转载）
    ]
    for pat in noise_suffixes:
        s = re.sub(pat, '', s).strip()

    # 5. 清理多余空格和符号
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.strip('\/\-｜|　 ')

    return s if s else title


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
            tag = tag.strip()
            if tag in KNOWN_CREATORS:
                return tag

    return ""


# ==========================================================================
# 阶段4：翻唱补全（在 aggregate_to_songs 时使用，此处仅预留）
# ==========================================================================

# 歌曲→P主映射表（从解析结果构建）
# song_creator_map = {song_name: original_creator}


# ==========================================================================
# 主解析函数
# ==========================================================================

def parse_title(title: str, tags: str = "") -> dict:
    """
    解析标题，提取结构化信息。

    输入:
        title: str  — 视频/分P标题
        tags:  str  — 逗号分隔的标签

    输出:
        dict  — {
            "song_name":         str,
            "original_creator":  str,
            "vocal_singer":      str,
            "is_reupload":       bool,
        }
    """
    if not title:
        return {"song_name": "", "original_creator": "", "vocal_singer": "", "is_reupload": False}

    # 阶段1
    bracket_info = extract_bracket_tokens(title)
    feat_creator = extract_feat_creator(title)
    slash_creator = extract_slash_creator(title)

    # 阶段2
    all_creators = bracket_info["creators"] + ([slash_creator] if slash_creator else [])
    song_name = extract_song_name(title, all_creators, bracket_info["vocal_singers"])

    # 阶段3
    original_creator = assign_creator(title, tags, bracket_info, feat_creator, slash_creator)

    # 歌姬
    vocal_singer = bracket_info["vocal_singers"][0] if bracket_info["vocal_singers"] else "IA"

    # 搬运标记
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
# 测试
# ==========================================================================

def main():
    df = pd.read_csv("data/ia_music_data.csv")
    df = df.drop_duplicates(subset=["bvid", "page"])

    # 取前 100 行 + 后 100 行测试
    test = pd.concat([df.head(100), df.tail(100)]).drop_duplicates()
    print(f"测试 {len(test)} 行\n")

    # 解析
    results = []
    for _, row in test.iterrows():
        r = parse_title(str(row["title"]), str(row.get("tags", "")))
        results.append(r)

    # 统计
    has_creator = sum(1 for r in results if r["original_creator"])
    has_song = sum(1 for r in results if r["song_name"] and r["song_name"] != str(test.iloc[results.index(r)]["title"]))
    reuploads = sum(1 for r in results if r["is_reupload"])

    print(f"P主 提取率: {has_creator}/{len(results)} ({has_creator/len(results):.0%})")
    print(f"歌名 提取率: {has_song}/{len(results)} ({has_song/len(results):.0%})")
    print(f"搬运 标记: {reuploads}")

    # Top P主
    creators = Counter(r["original_creator"] for r in results if r["original_creator"])
    print(f"\nTop 10 P主:")
    for name, n in creators.most_common(10):
        print(f"  {n:>3}  {name}")

    # Top 歌姬
    singers = Counter(r["vocal_singer"] for r in results)
    print(f"\n歌姬分布:")
    for name, n in singers.most_common(5):
        print(f"  {n:>3}  {name}")

    # 抽查 10 条
    print(f"\n{'='*80}")
    print("抽查 15 条")
    print(f"{'='*80}")
    for i in [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160]:
        if i >= len(results):
            break
        row = test.iloc[i]
        r = results[i]
        print(f"\n  原始: {row['title'][:70]}")
        print(f"  歌名: {r['song_name'][:60]}")
        print(f"  P主:  {r['original_creator'] or '(无)'}")
        print(f"  歌姬: {r['vocal_singer']}")
        if r["is_reupload"]:
            print(f"  搬运: True")


if __name__ == "__main__":
    main()
