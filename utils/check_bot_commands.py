import aiohttp
import json

from src.constants import (
    DISCORD_CLIENT_ID,
    DISCORD_BOT_TOKEN,
)

async def fetch_global_commands():
    headers = {
        "Authorization": f"Bot {DISCORD_BOT_TOKEN}",
        "Content-Type": "application/json"
    }

    url = f"https://discord.com/api/v10/applications/{DISCORD_CLIENT_ID}/commands"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            result = await response.json()
            return result

result = fetch_global_commands()

print(result)