print("Config package initialized.")

DEBUG = True

# MODEL ------------------------------------------------------------------------

USE_MODEL = 'Qwen'  # 「chatGPT， Qwen， tongyiQwen」

# OpenAI https://api.openai.com/v1/chat/completions
GPT_URL = 'https://api.openai.com/v1/chat/completions'
API_KEY = 'sk-xxxxxx'

# Qwen
Qwen_URL = 'http://127.0.0.1:8002'

# tongyiQwen
DASHSCOPE_API_KEY = "sk-xxxxxx"

# MODEL ------------------------------------------------------------------------

# CONFIGURATION ------------------------------------------------------------------------

# 意图相关性判断阈值0-1
RELATED_INTENT_THRESHOLD = 0.5

# CONFIGURATION ------------------------------------------------------------------------
