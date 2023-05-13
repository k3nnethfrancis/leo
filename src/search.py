# Description: This file contains the code for the question answering and onboarding systems for Use...
import os
import asyncio
import logging
import pandas as pd
import discord
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from src.constants import OPENAI_API_KEY, TARGET_CHANNEL_ID, BOT_INSTRUCTIONS, BOT_NAME, EXAMPLE_CONVOS
from src.moderation import moderate_message, send_moderation_flagged_message, send_moderation_blocked_message
from src.utils import split_into_shorter_messages, close_thread, logger
from src.base import BaseRetriever, Message, Prompt, Conversation
from src.moderation import send_moderation_flagged_message, send_moderation_blocked_message
import functools
import concurrent.futures
import asyncio
from typing import List

# ... Onboard.py document specific imports and code
LEO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# logger
logger = logging.getLogger(__name__)

# for listening to intros
# Put the ID of the Discord Channel you want the bot to respond to
devserve_LEO_LISTEN_CHANNEL_ID = 1094758337226215524  # Replace with your desired Channel ID

# Set bot name and example conversations from imported constants.
MY_BOT_NAME = BOT_NAME
MY_BOT_EXAMPLE_CONVOS = EXAMPLE_CONVOS

# Completion results for discord handling
class CompletionResult(Enum):
    OK = 0
    TOO_LONG = 1
    INVALID_REQUEST = 2
    OTHER_ERROR = 3
    MODERATION_FLAGGED = 4
    MODERATION_BLOCKED = 5

# dataclass for CompletionData for discord handling
@dataclass
class CompletionData:
    status: CompletionResult
    reply_text: Optional[str]
    status_text: Optional[str]

# Retriever class for searching embeddings db
class CustomRetriever(BaseRetriever):
    def search(self, query):
        results = super().search(query)
        return results

# Create an instance of CustomRetriever
retriever = CustomRetriever()

#### QA SYSTEM ####
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

## Onboarding Project Recommender System ##
# Template class for building prompts
class OnboardPromptTemplate:
    # initialize
    def __init__(self):
        self.examples_path = LEO_DIR + r'/data/csv/intro_examples.csv'
    
    # Load examples from csv for few shot learning
    def load_examples(self) -> dict:
        examples = pd.read_csv(self.examples_path)
        examples_dict = examples.to_dict('records')
        # return the examples
        return examples_dict
    @staticmethod
    def get_dynamic_prompt(examples_dict) -> FewShotPromptTemplate:
        # Next, we specify the template to format the examples we have provided.
        # We use the `PromptTemplate` class for this.
        example_formatter_template = """
        message: {message}
        class: {class}\n
        """
        example_prompt = PromptTemplate(
            input_variables=["message", "class"],
            template=example_formatter_template,
        )

        # We'll use the `LengthBasedExampleSelector` to select the examples.
        example_selector = LengthBasedExampleSelector(
            # These are the examples is has available to choose from.
            examples=examples_dict, 
            # This is the PromptTemplate being used to format the examples.
            example_prompt=example_prompt, 
            # This is the maximum length that the formatted examples should be.
            # Length is measured by the get_text_length function below.
            max_length=300,
        )
        # We can now use the `example_selector` to create a `FewShotPromptTemplate`.
        dynamic_prompt = FewShotPromptTemplate(
            # We provide an ExampleSelector instead of examples.
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix="Predict whether or not a message is an introduction.",
            suffix="message: {input}\nclass:",
            input_variables=["input"],
            example_separator="\n",
        )
        return dynamic_prompt
                     
                     
# initalize an instance of the OnboardPromptTemplate class
onboard_prompt_template_instance = OnboardPromptTemplate()
                     
class IntroDetector:
    def __init__(self):
        self.model = OpenAI(openai_api_key=OPENAI_API_KEY)
        onboard_prompt_template_instance = OnboardPromptTemplate()
        self.examples = onboard_prompt_template_instance.load_examples()
        self.prompt = onboard_prompt_template_instance.get_dynamic_prompt(self.examples)

    def is_intro(self, message: str) -> bool:
        prompt = self.prompt.format(input=message)
        response = self.model(prompt)
        classification_result = response.strip()
        return classification_result.lower() == "true"
    @staticmethod
    def intro2query(intro: str) -> str:
        return f"What are one or two projects that might be interesting for a user with the following intro: {intro}? \nPlease exlpained in a helpful tone."


# build out functionality of onboard command   
async def generate_onboard_completion_response(intro: List[str]
, user: str) -> CompletionData:
    # Process messages and ignore the ones from "leo-bot"
    inputs = [msg for msg in intro if not msg.startswith("leo-bot:")]
    inputs_str = "\n".join(inputs)
    
    # transform the intro to a query
    query = IntroDetector.intro2query(intro=inputs_str)


    logger.debug("Deploying OnboardBot to search for relevant projects...")

    # run the event loop in a thread pool to prevent blocking from discord
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # run the search in the thread pool with our query
        response = await loop.run_in_executor(executor, functools.partial(
            retriever.search,
            query=query,
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
async def process_onboard_response(user: str, interaction: discord.Interaction, message_id: int, response_data: CompletionData):
    status = response_data.status
    reply_text = response_data.reply_text
    status_text = response_data.status_text

    # Find the original message using the message ID
    target_channel = await interaction.client.fetch_channel(TARGET_CHANNEL_ID)
    original_message = await target_channel.fetch_message(message_id)

    if status is CompletionResult.OK or status is CompletionResult.MODERATION_FLAGGED:
        if reply_text:
            formatted_reply_text = f"Hey {user.mention}!\n\n{reply_text}"
            await original_message.reply(formatted_reply_text)
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