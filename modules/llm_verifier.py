"""
LLM 验证器：对正则分类标记为 suspicious 的行进行二次确认。
支持两个后端：
  - deepseek: DeepSeek V4 Flash（需 API Key，准确性高）
  - qwen:     Qwen2.5-1.5B QLoRA（本地免费，需预先训练 LoRA 权重）
"""

import json
import os
import re
import pandas as pd

# ==========================================================================
# Few-shot 示例（两个后端共用）
# ==========================================================================

FEWSHOT_EXAMPLES = [
    {"title": "【IA】六兆年と一夜物語【kemu】", "tags": "IA,VOCALOID,kemu", "category": "VOCALOID·UTAU", "copyright": "转载",
     "content_type": "ia_music", "is_game": False, "is_cover": False},
    {"title": "【IAオリジナル曲】花と記憶 / Orangestar", "tags": "IA,VOCALOID,Orangestar,IAオリジナル曲", "category": "VOCALOID·UTAU", "copyright": "自制",
     "content_type": "ia_music", "is_game": False, "is_cover": False},
    {"title": "【PJSK】Children Record【IA】", "tags": "PJSK,IA,VOCALOID", "category": "音游", "copyright": "转载",
     "content_type": "ia_music", "is_game": True, "is_cover": False},
    {"title": "【IAカバー】夜に駆ける / YOASOBI", "tags": "IA,翻唱,COVER,VOCALOID", "category": "VOCALOID·UTAU", "copyright": "自制",
     "content_type": "ia_music", "is_game": False, "is_cover": True},
    {"title": "【钢琴】六兆年と一夜物語【IA】", "tags": "IA,VOCALOID,钢琴,翻弹", "category": "演奏", "copyright": "自制",
     "content_type": "ia_music", "is_game": False, "is_cover": True},
    {"title": "【P主人物志】你所不知道的kemu/堀江晶太（上）", "tags": "GUMI,P主人物志,VOCALOID,IA,kemu", "category": "VOCALOID·UTAU", "copyright": "自制",
     "content_type": "ia_related", "is_game": False, "is_cover": False},
    {"title": "【VOCALOID】听开头猜术曲!(第二期)", "tags": "初音未来,VOCALOID,GUMI,猜歌,IA", "category": "VOCALOID·UTAU", "copyright": "自制",
     "content_type": "ia_related", "is_game": False, "is_cover": False},
    {"title": "最强虚拟歌姬争霸赛3", "tags": "v flower,IA,可不,gumi,虚拟歌手", "category": "VOCALOID·UTAU", "copyright": "自制",
     "content_type": "ia_related", "is_game": False, "is_cover": False},
    {"title": "战争雷霆：德系隐藏金币机-IA58普卡拉攻击机", "tags": "战争雷霆,游戏,IA,军事", "category": "单机游戏", "copyright": "自制",
     "content_type": "irrelevant", "is_game": False, "is_cover": False},
    {"title": "AI生成《恐龙快打》全角色真人化 | 作者：IA IA OH", "tags": "AI,游戏,恐龙快打", "category": "日常", "copyright": "自制",
     "content_type": "irrelevant", "is_game": False, "is_cover": False},
    {"title": "Roblox IA item asylum 实况", "tags": "IA,游戏,Roblox,itemasylum", "category": "单机游戏", "copyright": "自制",
     "content_type": "irrelevant", "is_game": False, "is_cover": False},
    {"title": "IA福袋 开箱", "tags": "生活记录,开箱,IA福袋", "category": "日常", "copyright": "自制",
     "content_type": "irrelevant", "is_game": False, "is_cover": False},
]

SYSTEM_PROMPT = (
    "你是一个B站视频分类助手。根据标题、标签、分区、版权信息，将视频分为3类：\n\n"
    "- ia_music: IA（VOCALOID/CeVIO歌姬）演唱的音乐作品。包括原创曲、翻唱、钢琴改编、合唱等\n"
    "- ia_related: 与IA相关但非歌曲投稿。语调教、P主介绍、猜歌比赛、MMD舞蹈、演唱会、科普等\n"
    "- irrelevant: 与虚拟歌姬IA完全无关。游戏实况、军事型号(IA58)、AI生成、开箱、漫剧等\n\n"
    "同时判断两个布尔标记：\n"
    "- is_game: 与音游相关（PJSK/バンドリ/Phigros/CHUNITHM/maimai/osu!/D4DJ等）\n"
    "- is_cover: 翻唱/改编（Cover/翻唱/カバー/钢琴版/演奏/混音等）\n\n"
    "输出JSON格式，只输出JSON不要解释。"
)


# ==========================================================================
# 解析 LLM 输出
# ==========================================================================

def _parse_result(raw_text: str, fallback: dict) -> dict:
    """从 LLM 原始文本中提取 JSON，失败时返回 fallback。"""
    text = raw_text.strip()
    if text.startswith("```"):
        # 移除 markdown fence，兼容单行和多行
        text = re.sub(r'^```\w*\s*', '', text)   # 开头 ``` 或 ```json
        text = re.sub(r'\s*```$', '', text)       # 结尾 ```

    try:
        result = json.loads(text)
        return {
            "content_type": result.get("content_type", fallback.get("content_type", "ia_music")),
            "is_game": result.get("is_game", fallback.get("is_game", False)),
            "is_cover": result.get("is_cover", fallback.get("is_cover", False)),
        }
    except json.JSONDecodeError:
        return fallback


# ==========================================================================
# 构建 prompt
# ==========================================================================

def _build_prompt(row: dict) -> str:
    """构建单条分类请求的 prompt 文本。"""
    parts = [
        SYSTEM_PROMPT,
        "",
        "示例：",
    ]
    for ex in FEWSHOT_EXAMPLES:
        parts.append(f"标题：{ex['title']}")
        parts.append(f"标签：{ex['tags']}")
        parts.append(f"分区：{ex['category']}")
        parts.append(f"版权：{ex['copyright']}")
        parts.append(f"→ {json.dumps({'content_type': ex['content_type'], 'is_game': ex['is_game'], 'is_cover': ex['is_cover']}, ensure_ascii=False)}")
        parts.append("")

    parts.append("现在分类：")
    parts.append(f"标题：{row.get('title', '')}")
    parts.append(f"标签：{row.get('tags', '')}")
    parts.append(f"分区：{row.get('category', '')}")
    parts.append(f"版权：{row.get('copyright', '')}")
    parts.append("→ ")

    return "\n".join(parts)


# ==========================================================================
# 后端1: DeepSeek V4 Flash
# ==========================================================================

class DeepSeekVerifier:
    def __init__(self, api_key: str = None):
        from openai import OpenAI
        from config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = DEEPSEEK_MODEL

    def verify_one(self, row: dict) -> dict:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in FEWSHOT_EXAMPLES:
            user_text = (
                f"标题：{ex['title']}\n标签：{ex['tags']}\n"
                f"分区：{ex['category']}\n版权：{ex['copyright']}"
            )
            assistant_text = json.dumps({
                "content_type": ex["content_type"],
                "is_game": ex["is_game"], "is_cover": ex["is_cover"],
            }, ensure_ascii=False)
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})

        user_text = (
            f"标题：{row.get('title', '')}\n标签：{row.get('tags', '')}\n"
            f"分区：{row.get('category', '')}\n版权：{row.get('copyright', '')}"
        )
        messages.append({"role": "user", "content": user_text})

        response = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=0.0, max_tokens=100,
        )
        fallback = {
            "content_type": row.get("content_type", "ia_music"),
            "is_game": row.get("is_game", False),
            "is_cover": row.get("is_cover", False),
        }
        return _parse_result(response.choices[0].message.content, fallback)


# ==========================================================================
# 后端2: Qwen2.5-1.5B QLoRA（本地）
# ==========================================================================

class QwenVerifier:
    """Qwen2.5-1.5B QLoRA 本地推理，使用与训练一致的 Alpaca 格式。"""

    def __init__(self, adapter_path: str = "models/content_classifier_lora"):
        import mlx_lm
        self.model, self.tokenizer = mlx_lm.load(
            "Qwen/Qwen2.5-1.5B-Instruct",
            adapter_path=adapter_path,
        )
        self.labels = ["ia_music", "ia_related", "irrelevant"]

    def verify_one(self, row: dict) -> dict:
        """分类单条，返回 content_type。is_game/is_cover 保持正则值。"""
        import mlx_lm
        # 与训练格式完全一致的 Alpaca prompt
        input_text = (
            f"标题：{row.get('title', '')}\n"
            f"标签：{row.get('tags', '')}\n"
            f"分区：{row.get('category', '')}"
        )
        prompt = (
            f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
            f"### Input:\n{input_text}\n\n### Response:\n"
        )

        result = mlx_lm.generate(
            self.model, self.tokenizer,
            prompt=prompt, max_tokens=5, verbose=False,
        )

        # 解析分类标签
        qwen_label = "ia_music"
        result_lower = result.lower().strip()
        for label in self.labels:
            if label in result_lower:
                qwen_label = label
                break

        # 保守策略：正则和 Qwen 在 music/related 间有分歧 → 归入 related
        # 保证 ia_music 绝对纯净（过气分析零污染）
        regex_label = row.get("content_type", "ia_music")
        if {regex_label, qwen_label} == {"ia_music", "ia_related"}:
            content_type = "ia_related"
        elif qwen_label == "ia_music" and regex_label != "ia_music":
            content_type = "ia_related"  # 正则没说是音乐，不相信Qwen的music判断
        else:
            content_type = qwen_label

        return {
            "content_type": content_type,
            "is_game": row.get("is_game", False),
            "is_cover": row.get("is_cover", False),
        }



# ==========================================================================
# 统一接口
# ==========================================================================

def verify_suspicious(df: pd.DataFrame, backend: str = "deepseek",
                      api_key: str = None, adapter_path: str = None) -> pd.DataFrame:
    """
    对 DataFrame 中 suspicious=True 的行逐一 LLM 验证，覆写分类结果。

    输入:
        df:           pd.DataFrame — 含 content_type/suspicious 列的完整数据
        backend:      "deepseek" | "qwen"
        api_key:      DeepSeek API Key（backend="deepseek" 时需要）
        adapter_path: QLoRA 适配器路径（backend="qwen" 时需要）

    输出:
        pd.DataFrame — LLM 覆写后的数据
    """
    suspicious_mask = df["suspicious"] == True
    suspicious_count = suspicious_mask.sum()
    if suspicious_count == 0:
        print("  [LLM] 无可疑条目，跳过")
        return df

    # 选择后端
    if backend == "deepseek":
        from config import DEEPSEEK_API_KEY
        key = api_key or DEEPSEEK_API_KEY or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            print("  [LLM] DeepSeek API Key 未配置，跳过 LLM 验证")
            df["llm_verified"] = False
            return df
        verifier = DeepSeekVerifier(key)
        label = "DeepSeek"
    elif backend == "qwen":
        path = adapter_path or "models/content_classifier_lora"
        if not os.path.exists(path):
            print(f"  [LLM] QLoRA 适配器不存在 ({path})，跳过 LLM 验证")
            df["llm_verified"] = False
            return df
        verifier = QwenVerifier(path)
        label = "Qwen"
    else:
        raise ValueError(f"未知后端: {backend}")

    print(f"  [LLM/{label}] 验证 {suspicious_count} 条可疑记录...")
    modified = 0

    for idx in df[suspicious_mask].index:
        row = df.loc[idx]
        try:
            result = verifier.verify_one(row)
            if result["content_type"] != row["content_type"]:
                modified += 1
            df.at[idx, "content_type"] = result["content_type"]
            df.at[idx, "is_game"] = result["is_game"]
            df.at[idx, "is_cover"] = result["is_cover"]
            df.at[idx, "suspicious"] = False
        except Exception as e:
            print(f"    LLM 验证失败 ({row.get('bvid', '?')}): {e}")

    df["llm_verified"] = True
    print(f"  [LLM/{label}] 完成：修改 {modified}/{suspicious_count} 条")
    return df
