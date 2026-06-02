"""标签统计，将 tags 分为音乐相关/无关，输出 tags.md。"""
import pandas as pd
from collections import Counter
import re

# ---- 加载 ----
df = pd.read_csv("data/ia_music_data.csv")
df = df.drop_duplicates(subset=["bvid", "page"])

all_tags = Counter()
for tags_str in df["tags"].dropna():
    for tag in tags_str.split(","):
        tag = tag.strip()
        if tag:
            all_tags[tag] += 1

# ---- 分类规则 ----
# 音乐相关关键词（大小写不敏感，部分匹配）
MUSIC_PATTERNS = [
    r'IA', r'VOCALOID', r'UTAU', r'CeVIO', r'ボカロ', r'vocaloid',
    r'歌', r'曲', r'音乐', r'音', r'楽', r'ミュージック', r'Music', r'MV', r'PV',
    r'翻唱', r'Cover', r'カバー', r'歌って', r'原创', r'オリジナル', r'Original',
    r'調教', r'調声', r'調', r'神调教',
    r'初音', r'ミク', r'Miku', r'镜音', r'リン', r'レン', r'Rin', r'Len',
    r'巡音', r'ルカ', r'Luka', r'GUMI', r'グミ', r'結月', r'ゆかり',
    r'MAYU', r'MEIKO', r'KAITO', r'洛天依', r'言和', r'可不', r'重音', r'テト',
    r'Fukase', r'Flower', r'VY1', r'VY2', r'ONE', r'Lily', r'Prima',
    r'伊爱', r'イア', r'IA[^a-zA-Z]',
    r'V家', r'VOCALOID', r'術力口', r'术力口', r'ボカロ',
    r'殿堂入', r'伝説入', r'殿堂',
    r'P主', r'プロデューサー',
    r'じん', r'Orangestar', r'orangestar', r'梅とら', r'まふまふ',
    r'r-906', r'Guiano', r'傘村', r'\*Luna', r'ねじ式', r'ATOLS',
    r'kemu', r'自然の敵', r'ササノマリイ', r'夏山',
    r'MMD', r'3D', r'PV', r'手描', r'手书',
    r'踊って', r'ダンス', r'踊', r'Choreography',
    r'ピアノ', r'吉他', r'演奏', r'Inst',
    r'歌姫', r'虚拟歌', r'ボーカル', r'シンガー',
    r'民族調', r'ROCK', r'ROCKS',  # IA ROCKS etc
    r'阳炎', r'カゲロウ', r'目隐',
    r'六兆年', r'哨戒班', r'黎明前线',
    r'サントラ', r'BGM', r'OST',
]


def is_music_tag(tag: str) -> bool:
    tag_upper = tag.upper()
    # 精确排除：纯数字、游戏术语、推广标签等
    if re.match(r'^[0-9]+$', tag):
        return False
    for pat in MUSIC_PATTERNS:
        if re.search(pat, tag_upper):
            return True
    return False


# ---- 分类 ----
related = []
unrelated = []
for tag, count in all_tags.items():
    if is_music_tag(tag):
        related.append((tag, count))
    else:
        unrelated.append((tag, count))

# ---- 输出 tags.md ----
with open("data/tags.md", "w", encoding="utf-8") as f:
    f.write("# Tags 分类统计\n\n")
    f.write(f"- 总记录数: {len(df)}\n")
    f.write(f"- 不同 tag 数: {len(all_tags)}\n")
    f.write(f"- 音乐相关: {len(related)}\n")
    f.write(f"- 无关: {len(unrelated)}\n\n")

    f.write("## 音乐相关 Tags\n\n")
    f.write("| 频次 | Tag |\n|------|-----|\n")
    for tag, count in related:
        f.write(f"| {count} | {tag} |\n")

    f.write("\n## 无关 Tags\n\n")
    f.write("| 频次 | Tag |\n|------|-----|\n")
    for tag, count in unrelated:
        f.write(f"| {count} | {tag} |\n")

print(f"音乐相关: {len(related)}, 无关: {len(unrelated)}")
print("已导出 → data/tags.md")
