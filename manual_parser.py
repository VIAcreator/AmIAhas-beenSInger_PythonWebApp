#!/usr/bin/env python3
"""手动标题解析辅助工具 —— 逐行浏览 music.csv，建立关键词→标准歌名+原唱者映射。"""

import os
import sys
import pandas as pd

# ── 路径配置 ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MUSIC_CSV = os.path.join(BASE, "data", "labeled_samples", "music.csv")
KEYWORD_CSV = os.path.join(BASE, "data", "song_keywords.csv")
PROGRESS_FILE = os.path.join(BASE, "data", ".parser_progress")

# ── 数据加载 ──────────────────────────────────────────────
df = pd.read_csv(MUSIC_CSV, on_bad_lines="skip", engine="python")
df = df.sort_values("play_count", ascending=False).reset_index(drop=True)

if os.path.exists(KEYWORD_CSV):
    mapping = pd.read_csv(KEYWORD_CSV)
else:
    mapping = pd.DataFrame(columns=["keyword", "song_name", "original_creator"])

start_row = 0
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        start_row = int(f.read().strip())

TOTAL = len(df)

# ── 辅助函数 ──────────────────────────────────────────────
def find_matches(title, mapping_df):
    """检查标题是否含已有的 keyword（大小写不敏感）。"""
    matches = []
    for _, row in mapping_df.iterrows():
        if row["keyword"].upper() in title.upper():
            matches.append(row.to_dict())
    return matches


def list_existing_songs(mapping_df):
    """列出已有歌曲（按歌名字母序），去重。"""
    songs = mapping_df[["song_name", "original_creator"]].drop_duplicates()
    songs = songs.sort_values("song_name")
    for i, (_, r) in enumerate(songs.iterrows()):
        print(f"  {i + 1}. {r['song_name']}  ({r['original_creator']})")
    return songs


def display_row(row, index):
    """展示当前行信息。"""
    bvid = row["bvid"]
    pc = row["play_count"]
    title = row["title"]
    tags = str(row["tags"])[:80]
    author = row.get("author", row.get("auther", ""))

    print(f"\n{'=' * 70}")
    print(f"[{index + 1}/{TOTAL}]  BV: {bvid}  |  播放: {pc:,}")
    print(f"标题: {title}")
    print(f"标签: {tags}")
    print(f"UP主: {author}")

    # 高亮【】内容
    import re
    highlighted = re.sub(r"(【[^】]+】)", r"\033[1;33m\1\033[0m", title)
    if highlighted != title:
        print(f"高亮: {highlighted}")

    matches = find_matches(title, mapping)
    if matches:
        print(f"\n  ⚡ 已有匹配:")
        for m in matches:
            print(f'     keyword="{m["keyword"]}" → {m["song_name"]}  by {m["original_creator"]}')
    return matches


def save_progress(idx):
    """保存映射和进度。"""
    mapping.to_csv(KEYWORD_CSV, index=False, encoding="utf-8-sig")
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(idx))
    print(f"\n  ✓ 已保存: {len(mapping)} 条映射, 进度 {idx}/{TOTAL}")


def new_song(title):
    """新建歌曲映射。"""
    print("\n输入关键词（逗号分隔，可从标题复制部分):")
    keywords = input("> ").split(",")
    keywords = [k.strip() for k in keywords if len(k.strip()) >= 2]

    if not keywords:
        print("  ✗ 没有有效关键词（至少2个字符）")
        return

    # 检查重复 keyword
    existing_kw = set(mapping["keyword"].str.lower())
    for kw in keywords:
        if kw.lower() in existing_kw:
            ans = input(f"  keyword '{kw}' 已存在，是否覆盖? [y/N]: ")
            if ans.lower() != "y":
                keywords.remove(kw)

    if not keywords:
        print("  ✗ 所有关键词已取消")
        return

    song_name = input("标准歌名: ").strip()
    creator = input("原唱者(P主): ").strip()

    if not song_name:
        print("  ✗ 歌名不能为空")
        return

    for kw in keywords:
        mapping.loc[len(mapping)] = [kw, song_name, creator]

    print(f"  ✓ 已添加 {len(keywords)} 个关键词 → {song_name}  by {creator}")


def add_to_existing(title):
    """将当前标题中的词加到已有歌曲。"""
    songs = list_existing_songs(mapping)
    if songs.empty:
        print("  暂无已有歌曲，请先 [n] 新建")
        return

    try:
        choice = int(input("\n选择编号: ")) - 1
        if choice < 0 or choice >= len(songs):
            print("  ✗ 无效编号")
            return
    except ValueError:
        print("  ✗ 请输入数字")
        return

    target = songs.iloc[choice]
    keyword = input("输入新关键词: ").strip()
    if len(keyword) < 2:
        print("  ✗ 关键词至少需要2个字符")
        return

    # 检查是否已存在
    if keyword.lower() in set(mapping["keyword"].str.lower()):
        print(f"  ✗ keyword '{keyword}' 已存在")
        return

    mapping.loc[len(mapping)] = [keyword, target["song_name"], target["original_creator"]]
    print(f"  ✓ 已添加: {keyword} → {target['song_name']}  by {target['original_creator']}")


def show_stats():
    """显示当前进度统计。"""
    labeled = len(mapping)
    unique_songs = mapping["song_name"].nunique()
    print(f"\n  📊 已标注: {labeled} 个关键词, {unique_songs} 首歌曲")
    print(f"  进度: {start_row}/{TOTAL} ({start_row / TOTAL * 100:.1f}%)")


# ── 主循环 ──────────────────────────────────────────────
print(f"\n  📁 已加载 {TOTAL} 行音乐数据, {len(mapping)} 条已有映射")
print(f"  📍 从第 {start_row + 1} 行继续\n")
print("  [s]跳过  [n]新建歌曲  [a]加到已有歌曲  [e]编辑当前行  [q]保存并退出  [stat]统计\n")

i = start_row
while i < TOTAL:
    row = df.iloc[i]
    matches = display_row(row, i)

    cmd = input("\n> ").strip().lower()

    if cmd == "s":
        i += 1

    elif cmd == "n":
        new_song(row["title"])
        # 新建后立即检查匹配是否更新
        new_matches = find_matches(row["title"], mapping)
        if new_matches:
            print(f"  ⚡ 当前标题现在匹配到 {len(new_matches)} 条映射")

    elif cmd == "a":
        add_to_existing(row["title"])

    elif cmd == "e":
        print(f"\n当前标题: {row['title']}")
        new_title = input("输入修正后的标题 (直接回车保持原样): ").strip()
        if new_title:
            df.at[df.index[i], "title"] = new_title
            print(f"  ✓ 标题已更新")

    elif cmd == "q":
        ans = input("确定保存并退出? [Y/n]: ").strip().lower()
        if ans in ("", "y"):
            save_progress(i)
            print("  再见!")
            sys.exit(0)

    elif cmd == "stat":
        show_stats()

    elif cmd == "undo":
        if len(mapping) > 0:
            removed = mapping.iloc[-1]
            mapping.drop(mapping.index[-1], inplace=True)
            print(f"  ✓ 已撤销: {removed['keyword']} → {removed['song_name']}")
        else:
            print("  ✗ 没有可撤销的操作")

    else:
        print(f"  ✗ 未知命令: '{cmd}'")
        print("  [s]跳过 [n]新建 [a]加到已有 [e]编辑 [q]退出 [stat]统计 [undo]撤销")
