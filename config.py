"""项目配置文件。敏感信息通过环境变量注入。"""
import os
from dotenv import load_dotenv

load_dotenv()

# Flask
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-production")

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 数据文件
DATA_DIR = "data"
BUILTIN_CSV = os.path.join(DATA_DIR, "ia_music_data.csv")
UPLOAD_DIR = "uploads"

# QLoRA 分类器
MODEL_DIR = "models"
LORA_PATH = os.path.join(MODEL_DIR, "content_classifier_lora")
LABELED_SAMPLES = os.path.join(DATA_DIR, "labeled_samples.json")
