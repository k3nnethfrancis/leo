import os
import asyncio
import logging
import discord
from typing import List
from langchain.schema import Document
from langchain.llms import OpenAI
from src.constants import OPENAI_API_KEY
from src.completion import CompletionResult, CompletionData
from src.docloader import get_loaded_documents, get_docsearch_instance, get_llm_instance

import os
import pandas as pd
from typing import List
from abc import ABC, abstractmethod
from langchain import OpenAI
from langchain.schema import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.document_loaders import DirectoryLoader, TextLoader
from src.constants import OPENAI_API_KEY


# get the parent directory of the current file
LEO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Initialize the OpenAI instance
# llm = OpenAI(openai_api_key=OPENAI_API_KEY)

class BaseRetriever(ABC):
    def __init__(self):
        path = LEO_DIR + r'/text/'
        self.loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
        self.index = VectorstoreIndexCreator().from_loaders([self.loader])
    @abstractmethod
    def search(self, query: str) -> List[Document]:
        """Responds to a query about the users documents.
            Args:
                query: string to find relevant docs for
            Returns:
                response to query
        """
        result = self.index.query(query)
        return result.split('\n')
    def get_relevant_projects(self, intro: str) -> List[str]:
        """Responds to a query about the users documents.
            Args:
                query: string to find relevant docs for
            Returns:
                response to query
        """
        query = f"What are one or two projects that might be interesting for a user with the following intro: {intro}? \nPlease respond in a helpful, welcoming tone."
        result = self.index.query(query)
        return result.split('\n') # Split the response text into a list of sentences

async def process_search_response(user: str, interaction: discord.Interaction, question: str, response_data: CompletionData):
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

