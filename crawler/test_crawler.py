"""
验证性爬虫：搜索10条视频 + 逐一获取详情，模拟批量爬取流程。
每次请求间隔 1~3s 随机值，验证安全性和返回数据结构。

使用方法：
    1. source venv/bin/activate
    2. python crawler/test_crawler.py
"""

import asyncio
import random
import re
import os
import pandas as pd
from datetime import datetime
from bilibili_api import search, video, request_settings
from bilibili_api.search import SearchObjectType, OrderVideo


# ==========================================================================
# 代理配置（可选）
# ==========================================================================
# 填入你的代理地址后取消注释即可启用。格式示例：
#   "http://用户名:密码@代理IP:端口"      # 付费代理（快代理/芝麻代理等）
#   "http://127.0.0.1:7890"              # 本地代理（Clash/V2Ray）

PROXY_URL = None

if PROXY_URL:
    request_settings.set_proxy(PROXY_URL)
    print(f"[代理] 已启用: {PROXY_URL.split('@')[-1] if '@' in PROXY_URL else PROXY_URL}")
else:
    print("[代理] 未配置，使用直连。请求间隔已设为 1~3s 随机，批量爬取时应不会被封。")


# ==========================================================================
# 爬取参数
# ==========================================================================
TEST_COUNT = 10                          # 本次测试爬取条数
SEARCH_KEYWORD = "IA"                    # 搜索关键词
MIN_DELAY = 1.0                          # 最小请求间隔（秒）
MAX_DELAY = 3.0                          # 最大请求间隔（秒）


def random_delay():
    """在 MIN_DELAY ~ MAX_DELAY 之间取随机值，降低被风控识别为机器人的概率。"""
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    print(f"  ⏳ 等待 {delay:.1f}s ...")
    return asyncio.sleep(delay)


# ==========================================================================
# 工具函数
# ==========================================================================

def clean_title(title: str) -> str:
    """移除搜索结果中 <em class="keyword"> 等HTML高亮标签。"""
    return re.sub(r'<[^>]+>', '', title)


def parse_duration_to_sec(duration_str: str) -> int:
    """
    将搜索API返回的 "MM:SS" 或 "HH:MM:SS" 格式转为秒数。
    详情API直接返回int秒数，搜索API返回字符串，这里做兜底转换。
    """
    parts = duration_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0


# ==========================================================================
# 阶段1：搜索获取 bvid 列表
# ==========================================================================

async def search_phase(count: int) -> list:
    """
    搜索并返回 bvid 列表。
    每次翻页之间间隔随机值，避免触发频率限制。
    """
    bvid_list = []
    page = 1

    print(f"\n{'='*60}")
    print(f"[阶段1] 搜索: keyword={SEARCH_KEYWORD}, order=CLICK")
    print(f"{'='*60}")

    while len(bvid_list) < count:
        await random_delay()

        result = await search.search_by_type(
            keyword=SEARCH_KEYWORD,
            search_type=SearchObjectType.VIDEO,
            order_type=OrderVideo.CLICK,
            page=page
        )

        items = result.get("result", [])
        if not items:
            print(f"  [提示] 第{page}页无更多结果，搜索结束。")
            break

        for item in items:
            bvid = item["bvid"]
            if bvid not in bvid_list:
                bvid_list.append(item)
            if len(bvid_list) >= count:
                break

        print(f"  第{page}页: 取到 {len(items)} 条，累计 {len(bvid_list)}/{count}")
        page += 1

    print(f"\n  搜索完成！共获取 {len(bvid_list)} 个唯一 bvid")
    return bvid_list


# ==========================================================================
# 阶段2：逐条获取详情 + 补全字段
# ==========================================================================

async def detail_phase(items: list) -> list:
    """
    对每条搜索结果调用 get_info()，补全搜索接口缺失的字段。
    每视频之间间隔随机值。
    """
    records = []

    print(f"\n{'='*60}")
    print(f"[阶段2] 获取详情: 共 {len(items)} 条")
    print(f"{'='*60}")

    for i, item in enumerate(items):
        bvid = item["bvid"]
        print(f"\n  [{i+1}/{len(items)}] bvid={bvid}  ", end="", flush=True)

        try:
            await random_delay()
            v = video.Video(bvid=bvid)
            info = await v.get_info()

            # 如果是多P视频，展开分P中标题含 "IA" 的页面
            pages = info.get("pages", [])
            videos_count = info["videos"]

            if videos_count <= 1:
                # --- 单P：一条记录 ---
                record = build_record(item, info, page_num=1, play_estimated=False)
                records.append(record)
                print(f"单P | title={record['title'][:30]}... | play={record['play_count']:,}")

            else:
                # --- 多P：筛选IA相关分P，均分播放量 ---
                ia_pages = [p for p in pages if is_ia_page(p.get("part", ""))]
                if not ia_pages:
                    print(f"多P({videos_count}P) 无IA分P，跳过")
                    continue

                per_page_plays = info["stat"]["view"] // videos_count
                for p in ia_pages:
                    record = build_record(
                        item, info,
                        page_num=p["page"],
                        page_title=p.get("part", ""),
                        play_count=per_page_plays,
                        play_estimated=True
                    )
                    records.append(record)
                print(f"多P({videos_count}P) → 取{len(ia_pages)}个IA分P | "
                      f"每P={per_page_plays:,}play | "
                      f"例: {ia_pages[0].get('part', '')[:30]}")

        except Exception as e:
            print(f"失败: {e}")

    print(f"\n  详情阶段完成！共生成 {len(records)} 条记录")
    return records


# ==========================================================================
# 工具函数
# ==========================================================================

def is_ia_page(part_title: str) -> bool:
    """
    判断分P标题是否标注 IA 演唱。
    匹配模式：-IA、-IA;ONE、feat.IA、feat. IA、【IA】等
    """
    # 匹配 "IA" 作为独立词汇出现（前后不接英文字母），排除 RADIAL、ASIA 等误匹配
    return bool(re.search(r'(?<![a-zA-Z])IA(?![a-zA-Z])', part_title))


def build_record(item: dict, info: dict, page_num: int = 1,
                 page_title: str = None, play_count: int = None,
                 play_estimated: bool = False) -> dict:
    """
    将搜索API返回(item)和详情API返回(info)合并为一条CSV记录。

    - 搜索API 提供：aid, bvid, mid, title(需清洗), author, pubdate, play,
                    video_review, review, favorites, tag, typename
    - 详情API 补全：duration(int), coin, share, like, tname, copyright
    - 多P视频：play_count 按均分计算，标 play_estimated=True
    """
    bvid = item["bvid"]
    title = page_title if page_title else clean_title(item["title"])
    raw_play = play_count if play_count is not None else item["play"]

    # 优先用详情API的 duration（int秒数），搜索API的 duration 是 "MM:SS" 字符串
    duration_sec = info.get("duration", 0)
    if duration_sec == 0:
        duration_sec = parse_duration_to_sec(item.get("duration", "0:00"))

    # tname 经常为空，用 typename（搜索API自带的分区名）作为兜底
    category = info.get("tname") or item.get("typename", "")

    return {
        "aid":            item["aid"],
        "bvid":           bvid,
        "page":           page_num,
        "mid":            item["mid"],
        "title":          title,
        "author":         item["author"],
        "publish_date":   datetime.fromtimestamp(item["pubdate"]),
        "duration_sec":   duration_sec,
        "play_count":     raw_play,
        "danmaku_count":  item.get("video_review", 0),
        "comment_count":  item.get("review", 0),
        "favorite_count": item.get("favorites", 0),
        "coin_count":     info.get("stat", {}).get("coin", 0),
        "share_count":    info.get("stat", {}).get("share", 0),
        "like_count":     info.get("stat", {}).get("like", 0),
        "tags":           item.get("tag", ""),
        "category":       category,
        "url":            f"https://www.bilibili.com/video/{bvid}",
        "copyright":      info.get("copyright", -1),
        "play_estimated": play_estimated,
    }


# ==========================================================================
# 主流程
# ==========================================================================

async def main():
    print("=" * 60)
    print(f"B站爬虫压力测试 — 搜索+详情，{TEST_COUNT}条")
    print(f"间隔: {MIN_DELAY}~{MAX_DELAY}s 随机")
    print("=" * 60)

    # 阶段1：搜索
    items = await search_phase(TEST_COUNT)
    if not items:
        print("[错误] 搜索无结果，退出。")
        return

    # 阶段2：逐条详情
    records = await detail_phase(items)

    # ======================================================================
    # 阶段3：导出 CSV
    # ======================================================================
    os.makedirs("data", exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv("data/test_batch.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"[导出] data/test_batch.csv")
    print(f"{'='*60}")
    print(f"  总记录: {len(df)} 条")
    print(f"  单P:    {(df['play_estimated'] == False).sum()} 条")
    print(f"  多P展开: {(df['play_estimated'] == True).sum()} 条（播放量为均分估算）")
    print(f"  列数:   {len(df.columns)}")
    print(f"  列名:   {', '.join(df.columns)}")

    # 显示采样
    print(f"\n  前 5 条预览:")
    print(df[["bvid", "page", "play_count", "play_estimated", "title"]]
          .head(5).to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
