from dotenv import load_dotenv
import os
import dacite
import yaml
from typing import Dict, List
from src.base import Config

# Load environment variables from the specified file
load_dotenv('/home/jovyan/leo/botenv.env')

# Read and load the configuration file
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG: Config = dacite.from_dict(
    Config, yaml.safe_load(open(os.path.join(SCRIPT_DIR, "config.yaml"), "r"))
)

# Extract bot-related information from the configuration file
BOT_NAME = CONFIG.name
BOT_INSTRUCTIONS = CONFIG.instructions
EXAMPLE_CONVOS = CONFIG.example_conversations

# Get Discord and OpenAI API keys from environment variables
DISCORD_BOT_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
DISCORD_CLIENT_ID = os.environ["DISCORD_CLIENT_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Read and process the allowed server IDs from environment variables
ALLOWED_SERVER_IDS: List[int] = []
server_ids = os.environ["ALLOWED_SERVER_IDS"].split(",")
for s in server_ids:
    ALLOWED_SERVER_IDS.append(int(s))

# Read and process server-to-moderation channel mapping from environment variables
SERVER_TO_MODERATION_CHANNEL: Dict[int, int] = {}
server_channels = os.environ.get("SERVER_TO_MODERATION_CHANNEL", "").split(",")
for s in server_channels:
    values = s.split(":")
    SERVER_TO_MODERATION_CHANNEL[int(values[0])] = int(values[1])

# Generate the bot invite URL with required permissions
BOT_INVITE_URL = f"https://discord.com/api/oauth2/authorize?client_id={DISCORD_CLIENT_ID}&permissions=328565073920&scope=bot"

# Set the moderation threshold values for blocked and flagged content
MODERATION_VALUES_FOR_BLOCKED = {
    "hate": 0.5,
    "hate/threatening": 0.1,
    "self-harm": 0.2,
    "sexual": 0.5,
    "sexual/minors": 0.2,
    "violence": 0.7,
    "violence/graphic": 0.8,
}

MODERATION_VALUES_FOR_FLAGGED = {
    "hate": 0.4,
    "hate/threatening": 0.05,
    "self-harm": 0.1,
    "sexual": 0.3,
    "sexual/minors": 0.1,
    "violence": 0.1,
    "violence/graphic": 0.1,
}

# Define constants related to bot behavior and messaging
SECONDS_DELAY_RECEIVING_MSG = 3  # Delay to catch multiple messages
MAX_THREAD_MESSAGES = 200  # Maximum number of messages in a thread
ACTIVATE_THREAD_PREFX = "💬✅"  # Prefix for active threads
INACTIVATE_THREAD_PREFIX = "💬❌"  # Prefix for inactive threads
MAX_CHARS_PER_REPLY_MSG = 1500  # Maximum characters per reply message (Discord limit: 2000)