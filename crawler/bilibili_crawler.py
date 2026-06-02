"""
B站 IA 音乐视频爬虫
====================
两阶段爬取：搜索 → 详情补全 + 多P展开
输出：data/ia_music_data.csv

使用方法：
    source venv/bin/activate
    python crawler/bilibili_crawler.py
"""

import asyncio
import random
import re
import os
import pandas as pd
from datetime import datetime
from typing import Optional

from bilibili_api import search, video, request_settings
from bilibili_api.search import SearchObjectType, OrderVideo


# ==========================================================================
# 配置区
# ==========================================================================

# 代理（可选）。格式：http://用户名:密码@IP:端口  或  http://127.0.0.1:7890
PROXY_URL: Optional[str] = None

# 请求间隔（秒），每次请求前在此范围内随机取值
MIN_DELAY = 1.0
MAX_DELAY = 3.0

# 目标数据量
TARGET_COUNT = 5000

# 第1档搜索配置：(关键词, 排序方式, 分区typeid, 标签)
SEARCH_CONFIGS = [
    ("IA",              OrderVideo.CLICK,   30, "IA/播放量/VOCALOID"),
    ("IA",              OrderVideo.PUBDATE, 30, "IA/最新/VOCALOID"),
    ("IA オリジナル曲",  OrderVideo.CLICK,   30, "IA原创曲/播放量/VOCALOID"),
    ("IA オリジナル曲",  OrderVideo.PUBDATE, 30, "IA原创曲/最新/VOCALOID"),
    ("IA 翻唱",          OrderVideo.CLICK,   30, "IA翻唱/播放量/VOCALOID"),
    ("IA 翻唱",          OrderVideo.PUBDATE, 30, "IA翻唱/最新/VOCALOID"),
]

# 第2档搜索配置（仅在第1档不足 TARGET_COUNT 时启用）
FALLBACK_CONFIGS = [
    ("IA", OrderVideo.CLICK,   None, "IA/播放量/全部分区"),
    ("IA", OrderVideo.PUBDATE, None, "IA/最新/全部分区"),
]

# 数据文件
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "ia_music_data.csv")

# CSV 列名（与架构设计文档一致）
CSV_COLUMNS = [
    "aid", "bvid", "page", "mid", "title", "author",
    "publish_date", "duration_sec", "play_count", "danmaku_count",
    "comment_count", "favorite_count", "coin_count", "share_count",
    "like_count", "tags", "category", "url", "copyright",
    "play_estimated",
]


# ==========================================================================
# 初始化
# ==========================================================================

def setup():
    """初始化代理和数据目录。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    if PROXY_URL:
        request_settings.set_proxy(PROXY_URL)


# ==========================================================================
# 工具函数
# ==========================================================================

async def random_delay():
    """每次请求前等待 1~3s 随机时长，降低被风控识别的概率。"""
    await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def clean_title(title: str) -> str:
    """移除 B站搜索结果的 <em class="keyword"> 等HTML高亮标签。"""
    return re.sub(r'<[^>]+>', '', title)


def is_ia_page(part_title: str) -> bool:
    """
    判断多P视频的某一P是否为 IA 演唱。
    使用单词边界匹配，排除 RADIAL、ASIA 等误匹配。
    """
    return bool(re.search(r'(?<![a-zA-Z])IA(?![a-zA-Z])', part_title))


def load_existing_keys(csv_path: str) -> tuple[set, set]:
    """
    从已有CSV中加载已爬取数据，返回两个集合：
      - known_pairs: {(bvid, page)} 复合键，用于详情阶段去重
      - known_bvids: {bvid} 唯一bvid，用于搜索阶段去重
    若CSV不存在则返回空集合。
    """
    if not os.path.exists(csv_path):
        return set(), set()

    df = pd.read_csv(csv_path, dtype={"bvid": str, "page": int})
    pairs = set(zip(df["bvid"], df["page"]))
    bvids = set(df["bvid"].unique())
    return pairs, bvids


# ==========================================================================
# 阶段1：搜索
# ==========================================================================

async def search_phase(known_bvids: set) -> list:
    """
    批量搜索，获取 bvid 列表及搜索级字段。
    每组配置翻页直到新增归零或达到 TARGET_COUNT。
    若第1档全部翻完后不足 TARGET_COUNT 则启用第2档。
    """
    new_videos = []

    def enough():
        return len(new_videos) >= TARGET_COUNT

    async def run_configs(configs, label_prefix):
        nonlocal new_videos
        for keyword, order, zone_id, label in configs:
            if enough():
                return
            page = 1
            print(f"  [{label_prefix}] {label}")
            while True:
                await random_delay()

                kwargs = {
                    "keyword": keyword,
                    "search_type": SearchObjectType.VIDEO,
                    "order_type": order,
                    "page": page,
                }
                if zone_id is not None:
                    kwargs["video_zone_type"] = zone_id

                result = await search.search_by_type(**kwargs)
                items = result.get("result", [])
                if not items:
                    print(f"    第{page}页: 0条 → 结束")
                    break

                new_count = 0
                for item in items:
                    bvid = item["bvid"]
                    # 用 (bvid, 1) 作为搜索阶段的临时键，仅用于搜索阶段去重
                    if bvid in known_bvids:
                        continue

                    record = {
                        "aid":            item["aid"],
                        "bvid":           bvid,
                        "mid":            item["mid"],
                        "title":          clean_title(item["title"]),
                        "author":         item["author"],
                        "publish_date":   datetime.fromtimestamp(item["pubdate"]),
                        "play_count":     item["play"],
                        "danmaku_count":  item["video_review"],
                        "comment_count":  item["review"],
                        "favorite_count": item["favorites"],
                        "tags":           item.get("tag", ""),
                        "category":       item.get("typename", ""),
                    }
                    new_videos.append(record)
                    known_bvids.add(bvid)
                    new_count += 1

                print(f"    第{page}页: {len(items)}条 → 新增{new_count}条 (累计{len(new_videos)}条)")
                # B站搜索翻到极限后不返回空，而是重复返回已有结果。
                # 若一页新增为0，说明已无新数据，终止该配置的翻页。
                if new_count == 0:
                    print(f"    新增归零，{label} 搜索提前终止")
                    break
                page += 1
                if enough():
                    print(f"    已达 {TARGET_COUNT} 条目标，停止搜索")
                    return
            print(f"    → {label} 翻页完毕，当前累计 {len(new_videos)} 条")
            if enough():
                return

    # 第1档
    await run_configs(SEARCH_CONFIGS, "1")
    print(f"  第1档完成：累计 {len(new_videos)} 条\n")

    # 第2档（兜底）
    if len(new_videos) < TARGET_COUNT:
        print(f"  不足 {TARGET_COUNT} 条，启用第2档（不限分区）")
        await run_configs(FALLBACK_CONFIGS, "2")
        print(f"  第2档完成：累计 {len(new_videos)} 条\n")

    return new_videos


# ==========================================================================
# 阶段2：详情补全 + 多P展开
# ==========================================================================

async def detail_phase(new_videos: list, known_pairs: set) -> list:
    """
    对每条搜索结果调用 get_info()，补全搜索缺失字段并处理多P。
    单P直接写入，多P筛选IA分P后均分播放量展开。
    """
    # 只处理前 TARGET_COUNT 条（搜索阶段可能超额收集）
    to_process = new_videos[:TARGET_COUNT]
    records = []
    total = len(to_process)

    for i, v_data in enumerate(to_process):
        bvid = v_data["bvid"]
        status = f"[{i+1}/{total}] {bvid}"

        try:
            await random_delay()
            v = video.Video(bvid=bvid)
            info = await v.get_info()

            videos_count = info["videos"]

            if videos_count <= 1:
                rec = build_record(v_data, info, page=1, play_estimated=False)
                key = (rec["bvid"], rec["page"])
                if key in known_pairs:
                    print(f"  {status} [跳过:已存在]")
                    continue
                records.append(rec)
                known_pairs.add(key)
                print(f"  {status} 单P | {rec['title'][:35]}")

            else:
                pages = info.get("pages", [])
                ia_pages = [p for p in pages if is_ia_page(p.get("part", ""))]

                if not ia_pages:
                    print(f"  {status} 多P({videos_count}P) 无IA分P，跳过")
                    continue

                per_page_plays = info["stat"]["view"] // videos_count

                skipped = 0
                for p in ia_pages:
                    rec = build_record(
                        v_data, info,
                        page=p["page"],
                        page_title=p.get("part", ""),
                        play_count=per_page_plays,
                        play_estimated=True,
                    )
                    key = (rec["bvid"], rec["page"])
                    if key in known_pairs:
                        skipped += 1
                        continue
                    records.append(rec)
                    known_pairs.add(key)

                print(f"  {status} 多P({videos_count}P)→取{len(ia_pages)}IA分P "
                      f"{'+跳过'+str(skipped) if skipped else ''}"
                      f"| 例:{ia_pages[0].get('part','')[:25]}")

        except Exception as e:
            print(f"  {status} 详情获取失败: {e}")

    return records


def build_record(search_item: dict, info: dict, page: int = 1,
                 page_title: str = None, play_count: int = None,
                 play_estimated: bool = False) -> dict:
    """
    合并搜索和详情数据，构造一条完整的CSV记录。
    """
    bvid = search_item["bvid"]

    # title：多P时用分P标题，单P时用搜索级标题
    title = page_title if page_title else search_item["title"]

    # play_count：多P时用均分值，单P时用搜索值
    if play_count is None:
        play_count = search_item["play_count"]

    # duration：优先用详情接口的 int 秒数，兜底用搜索接口的 "MM:SS" 字符串
    duration_sec = info.get("duration", 0)
    if duration_sec == 0:
        dur_str = search_item.get("duration", "0:00")
        parts = dur_str.split(":")
        if len(parts) == 3:
            duration_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            duration_sec = int(parts[0]) * 60 + int(parts[1])

    # category：搜索API的 typename 比详情API的 tname 更可靠
    category = search_item.get("category") or info.get("tname", "")

    # stat：安全取值
    stat = info.get("stat", {})

    return {
        "aid":            search_item["aid"],
        "bvid":           bvid,
        "page":           page,
        "mid":            search_item["mid"],
        "title":          title,
        "author":         search_item["author"],
        "publish_date":   search_item["publish_date"],
        "duration_sec":   duration_sec,
        "play_count":     play_count,
        "danmaku_count":  search_item.get("danmaku_count", 0),
        "comment_count":  search_item.get("comment_count", 0),
        "favorite_count": search_item.get("favorite_count", 0),
        "coin_count":     stat.get("coin", 0),
        "share_count":    stat.get("share", 0),
        "like_count":     stat.get("like", 0),
        "tags":           search_item.get("tags", ""),
        "category":       category,
        "url":            f"https://www.bilibili.com/video/{bvid}",
        "copyright":      info.get("copyright", -1),
        "play_estimated": play_estimated,
    }


# ==========================================================================
# 阶段3：写入CSV
# ==========================================================================

def save_to_csv(records: list):
    """将记录列表追加写入CSV，写入后自动去重。"""
    df_new = pd.DataFrame(records, columns=CSV_COLUMNS)
    file_exists = os.path.exists(CSV_PATH)

    if file_exists:
        # 追加模式：合并已有 + 新增，按 (bvid, page) 去重
        df_old = pd.read_csv(CSV_PATH, dtype={"bvid": str, "page": int})
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["bvid", "page"], keep="first")
        df_all.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    else:
        df_new.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")


# ==========================================================================
# 主流程
# ==========================================================================

async def main():
    """爬虫主入口。"""
    print("=" * 60)
    print("B站 IA 音乐视频爬虫")
    print("=" * 60)
    print()

    setup()
    if PROXY_URL:
        print(f"[代理] {PROXY_URL.split('@')[-1] if '@' in PROXY_URL else PROXY_URL}")
    else:
        print("[代理] 未配置，直连模式")
    print(f"[间隔] {MIN_DELAY}~{MAX_DELAY}s 随机")
    print(f"[目标] ≥ {TARGET_COUNT} 条")
    print()

    known_pairs, known_bvids = load_existing_keys(CSV_PATH)
    print(f"[去重] 已有 {len(known_pairs)} 条记录 ({len(known_bvids)} 个bvid)\n")

    # 阶段1
    print("[阶段1] 搜索")
    print("-" * 40)
    new_videos = await search_phase(known_bvids)
    print(f"\n[阶段1] 完成：发现 {len(new_videos)} 条新视频")
    if not new_videos:
        print("  无新数据，爬虫结束。")
        return

    # 阶段2
    print(f"\n[阶段2] 详情补全 + 多P展开")
    print("-" * 40)
    records = await detail_phase(new_videos, known_pairs)
    print(f"\n[阶段2] 完成：生成 {len(records)} 条记录")
    if not records:
        print("  无有效记录，爬虫结束。")
        return

    # 阶段3
    save_to_csv(records)

    # 统计
    estimated_count = sum(1 for r in records if r["play_estimated"])
    print(f"\n{'=' * 60}")
    print(f"爬取完成！")
    print(f"  新增记录:       {len(records)} 条")
    print(f"    单P:          {len(records) - estimated_count} 条")
    print(f"    多P展开(估算): {estimated_count} 条")
    print(f"  累计总计:       {len(known_pairs) + len(records)} 条")
    print(f"  输出文件:       {CSV_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
