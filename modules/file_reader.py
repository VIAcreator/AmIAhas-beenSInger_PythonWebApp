"""
模块1：文件读取与格式兼容
支持 CSV / Excel (.xlsx, .xls) / JSON 三种格式。
"""

import os
import pandas as pd
from flask import jsonify


# ==========================================================================
# 练习函数（由学生编写）
# ==========================================================================

def detect_format(filename: str) -> str:
    """
    根据文件扩展名判断文件格式。

    输入:
        filename: str  — 文件名，如 "data.csv"、"report.xlsx"、"output.json"

    输出:
        str  — "csv" | "excel" | "json"
        若扩展名不被识别，返回 "unknown"
    """
    # TODO: 取 filename 中最后一个 "." 之后的字符串，转为小写
    file_extension = filename.rsplit('.', 1)[-1].lower()
    if file_extension == 'csv':
        return "csv"
    elif file_extension == 'xls' or file_extension == 'xlsx':
        return "excel"
    elif file_extension == 'json':
        return "json"
    else:
        return "unknown"


def read_csv(path: str) -> pd.DataFrame:
    """
    读取 CSV 文件为 DataFrame。

    输入:
        path: str  — CSV 文件的完整路径，如 "data/ia_music_data.csv"

    输出:
        pd.DataFrame  — 文件内容
        CSV 第一行为列名，编码为 UTF-8
    """
    # TODO: 用 pd.read_csv() 读取，encoding="utf-8"
    readed_df = pd.read_csv(path, encoding="utf-8")
    return readed_df


def read_excel(path: str) -> pd.DataFrame:
    """
    读取 Excel 文件（.xlsx 或 .xls）为 DataFrame。

    输入:
        path: str  — Excel 文件的完整路径

    输出:
        pd.DataFrame  — 第一个 sheet 的内容
        Excel 第一行为列名
    """
    # TODO: 用 pd.read_excel() 读取
    readed_df = pd.read_excel(path)
    return readed_df


def read_json(path: str) -> pd.DataFrame:
    """
    读取 JSON 文件为 DataFrame。

    输入:
        path: str  — JSON 文件的完整路径
        JSON 格式：
        [
          {"col1": val, "col2": val, ...},
          {"col1": val, "col2": val, ...}
        ]

    输出:
        pd.DataFrame  — JSON 数组转为表格
    """
    # TODO: 用 pd.read_json() 读取
    readed_df = pd.read_json(path)
    return readed_df


# ==========================================================================
# 已编写部分（Flask 路由处理 + 调度逻辑）
# ==========================================================================

def read_file(path: str) -> pd.DataFrame:
    """
    自动识别文件格式并读取为 DataFrame。
    内部调用上面的三个练习函数。

    输入:
        path: str  — 文件路径

    输出:
        pd.DataFrame  — 文件内容

    异常:
        ValueError  — 文件格式不支持或文件不存在
    """
    if not os.path.exists(path):
        raise ValueError(f"文件不存在: {path}")

    fmt = detect_format(path)

    if fmt == "csv":
        return read_csv(path)
    elif fmt == "excel":
        return read_excel(path)
    elif fmt == "json":
        return read_json(path)
    else:
        raise ValueError(f"不支持的文件格式: {path}（支持 csv / xlsx / xls / json）")


def get_preview(df: pd.DataFrame, n: int = 20) -> dict:
    """
    返回 DataFrame 的前 N 行预览数据 + 列信息，供前端渲染表格。

    输入:
        df: pd.DataFrame  — 任意 DataFrame
        n:  int           — 预览行数，默认 20

    输出:
        dict  — {
            "columns":  ["col1", "col2", ...],          # 列名列表
            "dtypes":   {"col1": "int64", ...},         # 每列的 dtype
            "rows":     [[val, val, ...], ...],          # 前 n 行数据（每行是值列表）
            "row_count": 4128,                           # 总行数
            "col_count": 20                              # 总列数
        }

        rows 中的 datetime 类型需转为 ISO 格式字符串（用 pd.isnat() 判断 NaT）
    """
    preview = df.head(n)

    # 将 DataFrame 转为列表套列表，datetime 转为字符串
    rows = []
    for _, row in preview.iterrows():
        row_vals = []
        for val in row:
            if pd.isna(val):
                row_vals.append(None)
            elif hasattr(val, "isoformat"):
                row_vals.append(val.isoformat())
            else:
                row_vals.append(val)
        rows.append(row_vals)

    return {
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "rows": rows,
        "row_count": len(df),
        "col_count": len(df.columns),
    }


# ==========================================================================
# Flask 路由处理函数
# ==========================================================================

def handle_upload(request, get_df_fn, set_data_fn):
    """
    POST /upload 的处理函数。
    接收上传文件 → 读取 → 存入 g 对象 → 返回预览数据。

    request.files["file"] 为前端上传的文件对象。
    """
    if "file" not in request.files:
        return jsonify({"error": "未选择文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    # 保存到 uploads/ 目录
    path = os.path.join("uploads", file.filename)
    file.save(path)

    try:
        df = read_file(path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    set_data_fn(raw_df=df)
    preview = get_preview(df)

    return jsonify({"status": "ok", "filename": file.filename, "preview": preview})


def handle_preview(request, get_df_fn):
    """
    POST /api/preview 的处理函数。
    返回当前已加载数据的预览。
    """
    df = get_df_fn()
    if df is None:
        return jsonify({"error": "尚未加载数据"}), 400

    n = request.get_json(silent=True) or {}
    preview = get_preview(df, n=n.get("n", 20))
    return jsonify({"status": "ok", "preview": preview})
