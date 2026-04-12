import os
from dotenv import load_dotenv

load_dotenv()

# ═══ TELEGRAM ═══
BOT_TOKEN = os.getenv("BOT_TOKEN")

# ═══ AI KEYS ═══
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ═══ BOT SETTINGS ═══
BOT_NAME = "AssistX"
BOT_VERSION = "1.0.0"

# ═══ AI MODELS ═══
GROQ_MODEL = "llama3-70b-8192"      # Fast + Smart
GEMINI_MODEL = "gemini-1.5-flash"   # Free + Good

# ═══ DEFAULT SETTINGS ═══
DEFAULT_AI = "groq"                 # Default AI engine
MAX_HISTORY = 10                    # Kitne messages yaad rakhe
MAX_MESSAGE_LENGTH = 4096           # Telegram limit

# ═══ FREE PLAN LIMITS ═══
FREE_DAILY_MESSAGES = 20            # Free user ko 20 msg/day
FREE_IMAGE_GEN = 3                  # Free user ko 3 images/day

# ═══ ADMIN ═══
ADMIN_IDS = [5598017705]             # Apna Telegram ID daalo