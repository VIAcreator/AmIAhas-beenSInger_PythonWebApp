# B站 IA 视频分类标准 v2

## 三分类

| 类别 | 定义 | 典型示例 |
|------|------|---------|
| `ia_music` | IA（作为 VOCALOID/CeVIO 歌姬）演唱的音乐作品，包括原创、翻唱、音游曲、钢琴改编、合唱等 | `【IA】六兆年と一夜物語 / kemu` `【IAカバー】夜に駆ける` `【PJSK】Children Record【IA】` `【钢琴】六兆年と一夜物語` |
| `ia_related` | 与虚拟歌姬 IA 相关，但不是音乐作品投稿 | `P主人物志` `IA演唱会录像` `语调教` `猜歌比赛` `MMD舞蹈(IA模型)` `VOICEROID剧场` `声库评测` `周刊VOCALOID排行` `绘画过程` `纪录片` |
| `irrelevant` | 与虚拟歌姬 IA 完全无关的内容 | `战争雷霆 IA58飞机` `AI生成恐龙快打` `Roblox游戏实况` `装甲核心IA-C01B机甲` |

## 两个布尔标记

| 标记 | 含义 | 判定方式 |
|------|------|---------|
| `is_game` | 与音游相关 | 标题/tags/分区含音游关键词（PJSK/バンドリ/Phigros/CHUNITHM/osu! 等） |
| `is_cover` | 是翻唱（非IA原创曲） | 标题/tags 含 Cover/カバー/翻唱/歌ってみた/翻弹/合唱/演奏/日语版 等 |

### 标记如何参与筛选

```
过气分析 — 只筛选 ia_music
音游分层 — ia_music ∩ is_game=True  vs  ia_music ∩ is_game=False
原创vs翻唱 — ia_music ∩ is_cover=False  vs  ia_music ∩ is_cover=True
无关过滤 — content_type != 'irrelevant'
```

## 翻唱的特殊处理

### 问题

翻唱视频的标题通常**不标注原作者**（P主）。影响：
- `original_creator` 字段为空
- P主排行榜/过气分析中，翻唱的播放量无法归入对应 P主

### 解决方案

预处理管道中，`original_creator` 的提取优先级：

```
1. 标题含 "作曲/編曲/作詞/PV/Music" 等信息 → 直接提取
2. 标题含 "feat. XXX" → 提取 XXX
3. 标题含 "/ P主名" → 提取 P主名
4. tags 中匹配已知 P主列表 → 提取 P主名
5. 翻唱且以上皆无 → original_creator = ""（留空，不计入P主排行）
```

### 已知 P主列表

从 tags 统计中提取出现次数 > 10 的 P主名称：

```
Orangestar, jin(自然の敵P), kemu, 梅とら, まふまふ, r-906, Guiano,
傘村トータ, *Luna, ねじ式, ATOLS, うたたP, 150P, ギガP, ナナホシ管弦楽団,
花之祭P, じん, 自然の敵P, 神無月P, やいり, すいっち, VelecTi, ...
```

> 此列表随时间更新，存储在 `data/known_creators.json`。

## 原始数据筛选

预处理管道中，`content_type` 确定的流程：

```
1. 正则快速筛选（高置信度规则）
   - 音游关键词命中 → ia_music + is_game=True
   - 翻唱关键词命中 → ia_music + is_cover=True
   - tags 无 IA 且无 VOCALOID → irrelevant
   - 原创标记命中 → ia_music
   
2. 正则不确定 → 交 LLM（Qwen2.5-1.5B QLoRA 微调）
   - 输出 ia_music / ia_related / irrelevant
   - 同时输出 is_cover 布尔值

3. LLM 不可用 → 正则默认规则
   - category == VOCALOID·UTAU → ia_music
   - 否则 → ia_related
```

## 与 v1（五分类）的对比

| | v1 | v2 |
|---|----|----|
| 分类数 | 5 | 3 |
| 覆盖/翻唱 | 独立类别 | 布尔标记 |
| 音游 | 独立类别 | 布尔标记 |
| 无关/相关 | 分开 | 分开 |
| 预估准确率 | 70%（正则+LLM） | 85%+ |
| 过气分析 | 需手动合并 | 直接用 ia_music |
