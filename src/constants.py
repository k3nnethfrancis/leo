from dotenv import load_dotenv
import os
import dacite
import yaml
from typing import Dict, List
from src.base import Config

# get the parent directory of the current file
LEO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

load_dotenv(LEO_DIR+r'/botenv.env')

# Load the contents of config.yaml
# Get the script's directory
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG: Config = dacite.from_dict(
    Config, yaml.safe_load(open(os.path.join(SCRIPT_DIR, "config.yaml"), "r"))
)

# Retrieve the bot's name, instructions, and example conversations from the config
BOT_NAME = CONFIG.name
BOT_INSTRUCTIONS = CONFIG.instructions
EXAMPLE_CONVOS = CONFIG.example_conversations

# Retrieve API keys and tokens from environment variables
DISCORD_BOT_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
DISCORD_CLIENT_ID = os.environ["DISCORD_CLIENT_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Initialize an empty list to store allowed server IDs
ALLOWED_SERVER_IDS: List[int] = []
# Split the comma-separated string of server IDs from the environment variable
server_ids = os.environ["ALLOWED_SERVER_IDS"].split(",")
# Loop through the server IDs and append each as an integer to the allowed_server_ids list
for s in server_ids:
    ALLOWED_SERVER_IDS.append(int(s))

# Create a dictionary to map server IDs to their corresponding moderation channels
SERVER_TO_MODERATION_CHANNEL: Dict[int, int] = {}
# Retrieve the server_channels string from environment variables, or use an empty string if not present
server_channels = os.environ.get("SERVER_TO_MODERATION_CHANNEL", "").split(",")
# Loop through server_channels, splitting each pair on ':' and adding the pair to the dictionary
for s in server_channels:
    values = s.split(":")
    # Assign the mapped pair (server ID and moderation channel) to the dictionary
    SERVER_TO_MODERATION_CHANNEL[int(values[0])] = int(values[1])

# Create a Discord invite URL for the bot with specific permissions: 
#   # Send Messages, Create Public Threads, Send Messages in Threads, Manage Messages, Manage Threads, Read Message History, Use Slash Command
BOT_INVITE_URL = f"https://discord.com/api/oauth2/authorize?client_id={DISCORD_CLIENT_ID}&permissions=328565073920&scope=bot"

# Define a dictionary with moderation values to block messages based on categories
MODERATION_VALUES_FOR_BLOCKED = {
    "hate": 0.5,
    "hate/threatening": 0.1,
    "self-harm": 0.2,
    "sexual": 0.5,
    "sexual/minors": 0.2,
    "violence": 0.7,
    "violence/graphic": 0.8,
}

# Define a dictionary with moderation values to flag messages based on categories
MODERATION_VALUES_FOR_FLAGGED = {
    "hate": 0.4,
    "hate/threatening": 0.05,
    "self-harm": 0.1,
    "sexual": 0.3,
    "sexual/minors": 0.1,
    "violence": 0.1,
    "violence/graphic": 0.1,
}

# Set a delay in seconds for the bot's response
SECONDS_DELAY_RECEIVING_MSG = (
    3  # give a delay for the bot to respond so it can catch multiple messages
)

# Set a limit for the maximum number of messages in a thread
MAX_THREAD_MESSAGES = 200
# Define strings indicating activated and inactivated threads
ACTIVATE_THREAD_PREFX = "💬✅"
INACTIVATE_THREAD_PREFIX = "💬❌"
# Set a limit for the maximum number of characters in a single reply message
MAX_CHARS_PER_REPLY_MSG = (
    1500  # discord has a 2k limit, we just break message into 1.5k
)
