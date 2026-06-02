"""Flask 主入口。注册蓝图与路由，管理数据生命周期。"""
from flask import Flask, g, session, request, jsonify, render_template
import os

from config import SECRET_KEY, BUILTIN_CSV, UPLOAD_DIR


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 上传限 50MB

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    return app


app = create_app()


# ==========================================================================
# 数据生命周期管理
# ==========================================================================

def get_raw_df():
    """从 g 对象获取当前原始 DataFrame。若无则返回 None。"""
    return g.get("df_raw")


def get_clean_df():
    """从 g 对象获取清洗后的视频粒度 DataFrame。"""
    return g.get("df_clean")


def get_songs_df():
    """从 g 对象获取歌曲粒度 DataFrame。"""
    return g.get("df_songs")


def set_data(raw_df, clean_df=None, songs_df=None):
    """在 g 对象中存储 DataFrame。生命周期限于单次请求。"""
    g.df_raw = raw_df
    if clean_df is not None:
        g.df_clean = clean_df
    if songs_df is not None:
        g.df_songs = songs_df


# ==========================================================================
# 页面路由（HTML，暂存占位）
# ==========================================================================

@app.route("/")
def index():
    return "<h1>IA Music Analyzer</h1><p>首页 — 待开发</p>"


@app.route("/analysis")
def analysis():
    return "<h1>数据分析</h1><p>分析页 — 待开发</p>"


@app.route("/chat")
def chat():
    return "<h1>AI 问答</h1><p>问答页 — 待开发</p>"


# ==========================================================================
# API 路由（模块占位，Phase 2-6 逐步填充）
# ==========================================================================

@app.route("/upload", methods=["POST"])
def upload():
    """文件上传 API。"""
    from modules.file_reader import handle_upload
    return handle_upload(request, get_clean_df, set_data)


@app.route("/api/preview", methods=["POST"])
def api_preview():
    """数据预览 API。"""
    from modules.file_reader import handle_preview
    return handle_preview(request, get_clean_df)


@app.route("/api/clean", methods=["POST"])
def api_clean():
    """数据清洗 API。"""
    from modules.preprocessor import handle_clean
    return handle_clean(get_clean_df, set_data)


@app.route("/api/describe", methods=["POST"])
def api_describe():
    """描述统计 API。"""
    from modules.analyzer import handle_describe
    return handle_describe(request, get_clean_df, get_songs_df)


@app.route("/api/top", methods=["POST"])
def api_top():
    """排行榜 API。"""
    from modules.analyzer import handle_top
    return handle_top(request, get_clean_df, get_songs_df)


@app.route("/api/correlation", methods=["POST"])
def api_correlation():
    """相关性分析 API。"""
    from modules.analyzer import handle_correlation
    return handle_correlation(request, get_clean_df, get_songs_df)


@app.route("/api/trend", methods=["POST"])
def api_trend():
    """趋势分析 API。"""
    from modules.analyzer import handle_trend
    return handle_trend(request, get_clean_df, get_songs_df)


@app.route("/api/chart/<chart_type>", methods=["POST"])
def api_chart(chart_type):
    """图表生成 API。"""
    from modules.visualizer import handle_chart
    return handle_chart(chart_type, request, get_clean_df, get_songs_df)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """AI 问答 API。"""
    from modules.ai_assistant import handle_chat
    return handle_chat(request, get_clean_df, get_songs_df)


# ==========================================================================
# 启动
# ==========================================================================

if __name__ == "__main__":
    print("启动 IA Music Analyzer...")
    print(f"内置数据: {BUILTIN_CSV}")
    print(f"DeepSeek:  {'已配置' if __import__('config').DEEPSEEK_API_KEY else '未配置'}")
    app.run(debug=True, port=5000)
