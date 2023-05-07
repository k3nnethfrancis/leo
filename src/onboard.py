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
from src.constants import OPENAI_API_KEY, TARGET_CHANNEL_ID
import logging

# get the parent directory of the current file
LEO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))



#initialize the logger
logger = logging.getLogger(__name__)

# examples_df = pd.read_csv(LEO_DIR + r'/data/csv/intro_examples.csv')
# examples_dict = examples_df.to_dict('records')

class OnboardPromptTemplate:
    # initialize
    def __init__(self):
        self.examples_path = LEO_DIR + r'/data/csv/intro_examples.csv'
    
    # load the examples
    def load_examples(self) -> dict:
        examples = pd.read_csv(self.examples_path)
        examples_dict = examples.to_dict('records')
        # return the examples
        return examples_dict
    @staticmethod
    def get_dynamic_prompt(examples_dict) -> FewShotPromptTemplate:
        # All your existing code for creating and returning dynamic_prompt...
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

# Initialize the OpenAI instance
# llm = OpenAI(openai_api_key=OPENAI_API_KEY)

class IntroDetector:
    def __init__(self):
        self.model = OpenAI(openai_api_key=OPENAI_API_KEY)
        # self.examples = OnboardPromptTemplate.load_examples()
        # self.prompt = OnboardPromptTemplate.get_dynamic_prompt(self.examples)
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


import discord
import functools
import concurrent.futures
import asyncio
from typing import List, Optional
from enum import Enum
from dataclasses import dataclass
from src.base import BaseRetriever
from src.constants import (
    BOT_INSTRUCTIONS,
    BOT_NAME,
    EXAMPLE_CONVOS,
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


class CustomRetriever(BaseRetriever):
    def search(self, query):
        results = super().search(query)
        return results

# Create an instance of CustomRetriever
retriever = CustomRetriever()

# async def get_relevant_projects(self, intro: str) -> List[str]:
#         query = IntroDetector.intro2query(intro) #xform the intro into a prompt
#         result = retriever.search(query) #search for the prompt qa response
#         return result.split('\n')


async def generate_onboard_completion_response(intro: List[str]
, user: str) -> CompletionData:
    # Process messages and ignore the ones from "leo-bot"
    inputs = [msg for msg in intro if not msg.startswith("leo-bot:")]
    inputs_str = "\n".join(inputs)
    
    query = IntroDetector.intro2query(intro=inputs_str)
    
    # inputs = ["{}: {}".format("leo-bot" if intro.user == "leo-bot" else "user", intro.text) for intro in intro]
    # inputs_str = "\n".join(inputs)
    # query = IntroDetector.intro2query(inputs_str)


    logger.debug("Deploying OnboardBot to search for relevant projects...")

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
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