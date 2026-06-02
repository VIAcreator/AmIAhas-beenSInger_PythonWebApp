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

# 服务端缓存：{session_id: {"df_raw": DataFrame, "df_clean": DataFrame, "df_songs": DataFrame}}
# 简单字典实现，生产环境应换 Redis
_data_store = {}


def _get_store():
    """获取当前 session 对应的数据存储。"""
    sid = session.get("data_id")
    if not sid or sid not in _data_store:
        return None
    return _data_store[sid]


def get_raw_df():
    store = _get_store()
    return store["df_raw"] if store else None


def get_clean_df():
    store = _get_store()
    return store.get("df_clean") if store else None


def get_songs_df():
    store = _get_store()
    return store.get("df_songs") if store else None


def set_data(raw_df, clean_df=None, songs_df=None):
    """将 DataFrame 存入服务端缓存。"""
    import uuid
    sid = session.get("data_id")
    if not sid:
        sid = str(uuid.uuid4())[:8]
        session["data_id"] = sid
    if sid not in _data_store:
        _data_store[sid] = {}
    _data_store[sid]["df_raw"] = raw_df
    if clean_df is not None:
        _data_store[sid]["df_clean"] = clean_df
    if songs_df is not None:
        _data_store[sid]["df_songs"] = songs_df


# ==========================================================================
# 页面路由（HTML，暂存占位）
# ==========================================================================

@app.route("/")
def index():
    return render_template("index.html", page_title="数据加载")


@app.route("/analysis")
def analysis():
    return render_template("analysis.html", page_title="数据分析")


@app.route("/chat")
def chat():
    return render_template("chat.html", page_title="AI 问答")


# ==========================================================================
# API 路由（模块占位，Phase 2-6 逐步填充）
# ==========================================================================

@app.route("/upload", methods=["POST"])
def upload():
    """文件上传 API。"""
    from modules.file_reader import handle_upload
    return handle_upload(request, get_raw_df, set_data)


@app.route("/api/preview", methods=["POST"])
def api_preview():
    """数据预览 API。"""
    from modules.file_reader import handle_preview
    return handle_preview(request, get_raw_df)


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
    app.run(debug=True, port=5080)
