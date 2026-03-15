import os
from dotenv import load_dotenv

load_dotenv()

GRAPHHOPPER_API_KEY = os.getenv("GRAPHHOPPER_API_KEY")
KIE_API_KEY = os.getenv("KIE_API_KEY")
GRAPHHOPPER_BASE_URL = "https://graphhopper.com/api/1"
KIE_BASE_URL = "https://api.kie.ai"

# Redis
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Postgres
DATABASE_URL = os.getenv("DATABASE_URL")

# Auth0
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")

# Waze (via OpenWebNinja)
WAZE_API_KEY = os.getenv("WAZE_API_KEY")
WAZE_BASE_URL = "https://api.openwebninja.com/waze"

# ElevenLabs (STT + TTS)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # Default: George
