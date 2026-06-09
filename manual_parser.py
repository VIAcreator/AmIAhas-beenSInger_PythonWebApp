#!/usr/bin/env python3
"""手动标题解析辅助工具 —— 逐行浏览 music.csv，建立关键词→标准歌名+原唱者映射。"""

import os
import sys
import signal
import atexit
from datetime import datetime
import pandas as pd

# ── 路径配置 ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MUSIC_CSV = os.path.join(BASE, "data", "labeled_samples", "music.csv")
KEYWORD_CSV = os.path.join(BASE, "data", "song_keywords.csv")
PROGRESS_FILE = os.path.join(BASE, "data", ".parser_progress")
LOG_FILE = os.path.join(BASE, "data", ".parser_log.txt")

# ── 日志+信号：确保任何退出都保存进度 ──────────────────────
_log_dirty = False  # 标记是否有未保存的更改

def _log(msg):
    """同时输出到终端和日志文件。"""
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def _auto_save(idx):
    """自动保存（每次修改后调用），写入文件但不打印冗长信息。"""
    global _log_dirty
    try:
        mapping.to_csv(KEYWORD_CSV, index=False, encoding="utf-8-sig")
        with open(PROGRESS_FILE, "w") as f:
            f.write(str(idx))
        _log_dirty = False
    except Exception as e:
        _log(f"  ⚠ 自动保存失败: {e}")

def _emergency_save(idx):
    """紧急保存 —— 进程退出前最后一道防线。"""
    global _log_dirty
    try:
        mapping.to_csv(KEYWORD_CSV, index=False, encoding="utf-8-sig")
        with open(PROGRESS_FILE, "w") as f:
            f.write(str(idx))
        _log(f"\n  ⚡ 紧急保存完成: {len(mapping)} 条映射, 进度 {idx}/{TOTAL}")
        _log_dirty = False
    except Exception:
        pass  # 紧急保存失败就真的没办法了

def _signal_handler(sig, frame, current_idx):
    """Ctrl+C 或 SIGTERM 时触发保存。"""
    _log(f"\n\n  ⚠ 收到终止信号 (signal={sig})，正在紧急保存...")
    _emergency_save(current_idx)
    sys.exit(1)

# 注册 atexit 兜底 —— 用 [0] 占位，start_row 定义后更新
_atexit_save_idx = [0]

def _atexit_handler():
    if _log_dirty:
        _emergency_save(_atexit_save_idx[0])

atexit.register(_atexit_handler)

# 初始化日志文件（追加模式，标注启动时间）
with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ── Tee: 将所有 print 输出同时写入日志文件 ──────────────────
class Tee:
    """同时写入 stdout 和日志文件。"""
    def __init__(self, log_path):
        self.stdout = sys.stdout
        self.log = open(log_path, "a", encoding="utf-8")

    def write(self, data):
        self.stdout.write(data)
        self.log.write(data)

    def flush(self):
        self.stdout.flush()
        self.log.flush()

sys.stdout = Tee(LOG_FILE)

# ── 数据加载 ──────────────────────────────────────────────
df = pd.read_csv(MUSIC_CSV, on_bad_lines="skip")

# 探测实际最大列数（表头只有 11 列，但 950 行有 13 列，被 on_bad_lines 跳过了）
import csv
max_cols = 0
with open(MUSIC_CSV, encoding="utf-8") as f:
    for row in csv.reader(f):
        max_cols = max(max_cols, len(row))
# 构造足够宽的列名，重新读取全量
col_names = pd.read_csv(MUSIC_CSV, nrows=0).columns.tolist()
while len(col_names) < max_cols:
    col_names.append(f"_extra_{len(col_names)}")
df = pd.read_csv(MUSIC_CSV, names=col_names, skiprows=1, on_bad_lines="skip")
df = df.sort_values("play_count", ascending=False).reset_index(drop=True)

if os.path.exists(KEYWORD_CSV):
    mapping = pd.read_csv(KEYWORD_CSV)
else:
    mapping = pd.DataFrame(columns=["keyword", "song_name", "original_creator"])

start_row = 0
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        start_row = int(f.read().strip())

_atexit_save_idx[0] = start_row

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


def new_song(title, current_idx):
    """新建歌曲映射。"""
    global _log_dirty
    print("\n输入关键词（逗号分隔，可从标题复制部分):")
    keywords = input("> ").split(",")
    keywords = [k.strip() for k in keywords if len(k.strip()) >= 1]

    if not keywords:
        print("  ✗ 没有有效关键词（至少1个字符）")
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

    _log_dirty = True
    _auto_save(current_idx)
    print(f"  ✓ 已添加 {len(keywords)} 个关键词 → {song_name}  by {creator}")


def add_to_existing(title, current_idx):
    """将当前标题中的词加到已有歌曲。"""
    global _log_dirty
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
    if len(keyword) < 1:
        print("  ✗ 关键词至少需要1个字符")
        return

    # 检查是否已存在
    if keyword.lower() in set(mapping["keyword"].str.lower()):
        print(f"  ✗ keyword '{keyword}' 已存在")
        return

    mapping.loc[len(mapping)] = [keyword, target["song_name"], target["original_creator"]]
    _log_dirty = True
    _auto_save(current_idx)
    print(f"  ✓ 已添加: {keyword} → {target['song_name']}  by {target['original_creator']}")


def export_progress(current_idx):
    """导出当前进度为 CSV 文件。"""
    print("\n导出选项:")
    print("  [1] 仅导出 song_keywords 映射表")
    print("  [2] 导出 music.csv + 匹配结果（标注进度报告）")
    print("  [3] 两者都导出")
    choice = input("> ").strip()

    if choice not in ("1", "2", "3"):
        print("  ✗ 无效选项")
        return

    if choice in ("1", "3"):
        mapping.to_csv(KEYWORD_CSV, index=False, encoding="utf-8-sig")
        print(f"  ✓ 映射表已保存: {KEYWORD_CSV} ({len(mapping)} 条)")

    if choice in ("2", "3"):
        # 为已遍历过的行添加匹配列
        export_df = df.iloc[: current_idx].copy()
        matched_songs = []
        matched_creators = []
        for _, row in export_df.iterrows():
            matches = find_matches(row["title"], mapping)
            if matches:
                matched_songs.append(" | ".join(str(m["song_name"]) for m in matches))
                matched_creators.append(" | ".join(str(m["original_creator"]) for m in matches))
            else:
                matched_songs.append("")
                matched_creators.append("")

        export_df["matched_song"] = matched_songs
        export_df["matched_creator"] = matched_creators

        report_path = os.path.join(BASE, "data", "parse_progress_report.csv")
        export_df.to_csv(report_path, index=False, encoding="utf-8-sig")
        matched_count = sum(1 for s in matched_songs if s)
        print(f"  ✓ 进度报告已保存: {report_path}")
        print(f"    共 {len(export_df)} 行, 其中 {matched_count} 行已匹配, {len(export_df) - matched_count} 行待标注")


def modify_song(current_idx):
    """修改已标注歌曲的标准歌名或原唱者。"""
    global _log_dirty
    songs = list_existing_songs(mapping)
    if songs.empty:
        print("  暂无已有歌曲")
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
    old_name = target["song_name"]
    old_creator = target["original_creator"]

    print(f"\n当前: {old_name}  by {old_creator}")
    new_name = input(f"新歌名 (直接回车保留 '{old_name}'): ").strip()
    new_creator = input(f"新P主 (直接回车保留 '{old_creator}'): ").strip()

    if not new_name and not new_creator:
        print("  ✗ 未做任何修改")
        return

    final_name = new_name if new_name else old_name
    final_creator = new_creator if new_creator else old_creator

    # 更新所有匹配的行
    mask = (mapping["song_name"] == old_name) & (mapping["original_creator"] == old_creator)
    count = mask.sum()
    mapping.loc[mask, "song_name"] = final_name
    mapping.loc[mask, "original_creator"] = final_creator
    _log_dirty = True
    _auto_save(current_idx)
    print(f"  ✓ 已更新 {count} 条映射: {final_name}  by {final_creator}")


def alias_song(current_idx):
    """查看并管理歌曲的别名（关键词）。"""
    global _log_dirty
    songs = list_existing_songs(mapping)
    if songs.empty:
        print("  暂无已有歌曲")
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
    name = target["song_name"]
    creator = target["original_creator"]

    while True:
        # 获取该歌曲的所有关键词
        mask = (mapping["song_name"] == name) & (mapping["original_creator"] == creator)
        kws = mapping.loc[mask, "keyword"].tolist()
        print(f"\n  📌 {name}  by {creator}")
        print(f"  别名 ({len(kws)} 个): {', '.join(kws)}")
        print("  [a]添加别名  [d]删除别名  [b]返回")
        sub = input("> ").strip().lower()

        if sub == "b":
            break

        elif sub == "a":
            new_kw = input("  新别名: ").strip()
            if len(new_kw) < 1:
                print("  ✗ 别名不能为空")
                continue
            if new_kw.lower() in set(mapping["keyword"].str.lower()):
                print(f"  ✗ '{new_kw}' 已存在")
                continue
            mapping.loc[len(mapping)] = [new_kw, name, creator]
            _log_dirty = True
            _auto_save(current_idx)
            print(f"  ✓ 已添加: {new_kw}")

        elif sub == "d":
            if len(kws) <= 1:
                print("  ✗ 至少保留一个别名，如需删除整首歌曲请用其他方式")
                continue
            for j, kw in enumerate(kws):
                print(f"  {j + 1}. {kw}")
            try:
                del_choice = int(input("  删除编号: ")) - 1
                if del_choice < 0 or del_choice >= len(kws):
                    print("  ✗ 无效编号")
                    continue
            except ValueError:
                print("  ✗ 请输入数字")
                continue
            kw_to_del = kws[del_choice]
            # 只删除该歌曲下的该关键词
            del_mask = mask & (mapping["keyword"] == kw_to_del)
            mapping.drop(mapping[del_mask].index, inplace=True)
            _log_dirty = True
            _auto_save(current_idx)
            print(f"  ✓ 已删除: {kw_to_del}")

        else:
            print(f"  ✗ 未知命令: '{sub}'")


def show_stats():
    """显示当前进度统计。"""
    labeled = len(mapping)
    unique_songs = mapping["song_name"].nunique()
    print(f"\n  📊 已标注: {labeled} 个关键词, {unique_songs} 首歌曲")
    print(f"  进度: {start_row}/{TOTAL} ({start_row / TOTAL * 100:.1f}%)")


# ── 主循环 ──────────────────────────────────────────────
print(f"\n  📁 已加载 {TOTAL} 行音乐数据, {len(mapping)} 条已有映射")
print(f"  📍 从第 {start_row + 1} 行继续\n")
print("  [s]跳过  [n]新建歌曲  [a]加到已有歌曲  [k]别名管理  [m]修改歌曲  [e]编辑当前行  [q]保存并退出  [stat]统计  [exp]导出\n")

i = start_row
_atexit_save_idx[0] = i

# 安装信号处理器 —— 用闭包捕获 i
def _make_handler():
    def handler(sig, frame):
        _signal_handler(sig, frame, _atexit_save_idx[0])
    return handler

signal.signal(signal.SIGINT, _make_handler())
signal.signal(signal.SIGTERM, _make_handler())

try:
    while i < TOTAL:
        _atexit_save_idx[0] = i
        row = df.iloc[i]
        matches = display_row(row, i)

        cmd = input("\n> ").strip().lower()

        if cmd == "s":
            i += 1
            # 每 10 行自动保存进度
            if i % 10 == 0:
                _auto_save(i)

        elif cmd == "n":
            new_song(row["title"], i)

        elif cmd == "a":
            add_to_existing(row["title"], i)

        elif cmd == "m":
            modify_song(i)

        elif cmd == "k":
            alias_song(i)

        elif cmd == "e":
            print(f"\n当前标题: {row['title']}")
            new_title = input("输入修正后的标题 (直接回车保持原样): ").strip()
            if new_title:
                df.at[df.index[i], "title"] = new_title
                _log_dirty = True
                _auto_save(i)
                print(f"  ✓ 标题已更新")

        elif cmd == "q":
            ans = input("确定保存并退出? [Y/n]: ").strip().lower()
            if ans in ("", "y"):
                save_progress(i)
                _log_dirty = False
                print("  再见!")
                sys.exit(0)

        elif cmd == "stat":
            show_stats()

        elif cmd == "exp":
            export_progress(i)

        elif cmd == "undo":
            if len(mapping) > 0:
                removed = mapping.iloc[-1]
                mapping.drop(mapping.index[-1], inplace=True)
                _log_dirty = True
                _auto_save(i)
                print(f"  ✓ 已撤销: {removed['keyword']} → {removed['song_name']}")
            else:
                print("  ✗ 没有可撤销的操作")

        else:
            print(f"  ✗ 未知命令: '{cmd}'")
            print("  [s]跳过 [n]新建 [a]加到已有 [k]别名 [m]修改 [e]编辑 [q]退出 [stat]统计 [exp]导出 [undo]撤销")

finally:
    # finally 是最后一道防线：无论如何退出都尝试保存
    _atexit_save_idx[0] = i
    if _log_dirty:
        _emergency_save(i)


# ── 标注完成后的管理模式 ──────────────────────────────────
if i >= TOTAL:
    print(f"\n  🎉 全部 {TOTAL} 行已浏览完毕!")
    print("  进入管理模式，支持: [n]新建歌曲 [k]别名管理 [m]修改歌曲 [stat]统计 [exp]导出 [q]退出\n")
    while True:
        cmd = input("> ").strip().lower()

        if cmd == "q":
            ans = input("确定退出? [Y/n]: ").strip().lower()
            if ans in ("", "y"):
                save_progress(i)
                _log_dirty = False
                print("  再见!")
                break

        elif cmd == "n":
            title = input("标题（用于关键词匹配测试，可留空）: ").strip()
            new_song(title, i)

        elif cmd == "k":
            alias_song(i)

        elif cmd == "m":
            modify_song(i)

        elif cmd == "stat":
            show_stats()

        elif cmd == "exp":
            export_progress(i)

        elif cmd == "undo":
            if len(mapping) > 0:
                removed = mapping.iloc[-1]
                mapping.drop(mapping.index[-1], inplace=True)
                _log_dirty = True
                _auto_save(i)
                print(f"  ✓ 已撤销: {removed['keyword']} → {removed['song_name']}")
            else:
                print("  ✗ 没有可撤销的操作")

        else:
            print(f"  ✗ 未知命令: '{cmd}'")
            print("  [n]新建 [k]别名 [m]修改 [stat]统计 [exp]导出 [q]退出")
