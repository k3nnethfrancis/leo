# this is the file we will build the question answering system functionality into

"""Ask a question to the bot about the talentDAO database."""
import os
import asyncio
import logging
import discord
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
#from src.completion import CompletionResult, CompletionData
from src.constants import OPENAI_API_KEY
# Description: This file contains the code for generating a response from the OpenAI API
# Import necessary libraries and modules: enum, dataclass, openai, various functions, and constants from other files.
from enum import Enum
from dataclasses import dataclass
import logging
import openai
from src.moderation import moderate_message
from typing import Optional, List, Dict, Any
from src.constants import (
    BOT_INSTRUCTIONS,
    BOT_NAME,
    EXAMPLE_CONVOS,
)
import discord
from src.base import Message, Prompt, Conversation
from src.utils import split_into_shorter_messages, close_thread, logger
from src.moderation import (
    send_moderation_flagged_message,
    send_moderation_blocked_message,
)

# for listening to intros
# Put the ID of the Discord Channel you want the bot to respond to
devserve_LEO_LISTEN_CHANNEL_ID = 1094758337226215524  # Replace with your desired Channel ID

# Set bot name and example conversations from imported constants.
MY_BOT_NAME = BOT_NAME
MY_BOT_EXAMPLE_CONVOS = EXAMPLE_CONVOS

# Create an Enum to represent different completion results a message can have: 
#   # OK, too long, invalid request, other error, or whether it was flagged or blocked by moderation.
class CompletionResult(Enum):
    OK = 0
    TOO_LONG = 1
    INVALID_REQUEST = 2
    OTHER_ERROR = 3
    MODERATION_FLAGGED = 4
    MODERATION_BLOCKED = 5

# Define a new dataclass named CompletionData
@dataclass
class CompletionData:
    status: CompletionResult
    reply_text: Optional[str]
    status_text: Optional[str]


import functools
import concurrent.futures
import asyncio
from typing import List

#initialize the logger
logger = logging.getLogger(__name__)

# Load the documents and components needed for the QA system
# get the parent directory of the current file
LEO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# doc_dir = LEO_DIR+r'/text'

from src.base import BaseRetriever

class CustomRetriever(BaseRetriever):
    def search(self, query):
        results = super().search(query)
        return results

# Create an instance of CustomRetriever
retriever = CustomRetriever()


async def generate_qa_completion_response(query: List[str]
, user: str) -> CompletionData:
    inputs = ["{}: {}".format("Leo" if query.user == "Leo" else "user", query.text) for query in query]
    inputs_str = "\n".join(inputs)

    logger.debug("Deploying BaseRetriever to search for answer...")

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        response = await loop.run_in_executor(executor, functools.partial(
            retriever.search,
            query=inputs_str,
        ))
    response_text = response[0]  # Get the first element from the 'response' list
    logger.debug("Received response from OpenAI API")
    response_data = CompletionData(
        status=CompletionResult.OK,
        reply_text=response_text,
        status_text=None
    )
    return response_data


### Process the response from discord handling
async def process_qa_response(user: str, interaction: discord.Interaction, question: str, response_data: CompletionData):
    status = response_data.status
    reply_text = response_data.reply_text
    status_text = response_data.status_text

    if status is CompletionResult.OK or status is CompletionResult.MODERATION_FLAGGED:
        if reply_text:
            formatted_reply_text = f"**Question:** {question}\n\n**{user.mention}, here's the answer:**\n\n{reply_text}"
            await interaction.edit_original_response(content=formatted_reply_text)
        else:
            await interaction.followup.send(content="No response generated.", ephemeral=True)

        if status is CompletionResult.MODERATION_FLAGGED:
            await interaction.channel.send(
                f"⚠️ **This conversation has been flagged by moderation.**"
            )
    elif status is CompletionResult.MODERATION_BLOCKED:
        await interaction.channel.send(
            f"❌ **The response has been blocked by moderation.**"
        )

    elif status is CompletionResult.TOO_LONG or status is CompletionResult.INVALID_REQUEST or status is CompletionResult.OTHER_ERROR:
        await interaction.followup.send(
            content=f"**Error** - {status_text}", ephemeral=True
        )