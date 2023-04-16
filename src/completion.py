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


async def generate_completion_response(
    messages: List[Message], user: str
) -> CompletionData:
    '''Generate a completion response from OpenAI's API.
    
    Args:
        messages (List[Message]): A list of Message objects representing the conversation history.
        user (str): The user's Discord ID.
        
    Returns:
        CompletionData: A CompletionData object containing the completion result, reply text, and status text.
    '''
    try:
        # Create a Prompt instance using bot information and messages provided
        prompt = Prompt(
            header=Message(
                "System", f"Instructions for {MY_BOT_NAME}: {BOT_INSTRUCTIONS}" # BOT_INSTRUCTIONS is imported from constants.py
            ),
            examples=MY_BOT_EXAMPLE_CONVOS, # MY_BOT_EXAMPLE_CONVOS is imported from constants.py
            convo=Conversation(messages + [Message(MY_BOT_NAME)]), # Conversation is imported from base.py
        )
        # Generate the prompt text from the created Prompt instance
        rendered = prompt.render() # Prompt.render() is defined in base.py
        # Call OpenAI's API to create a completion using the generated text
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=rendered,
            temperature=1.0,
            top_p=0.9,
            max_tokens=512,
            stop=["<|endoftext|>"],
        )
        # Extract and clean the reply text from the generated response
        reply = response.choices[0].text.strip()
        if reply:
            # Moderate the response message using the moderation framework
            flagged_str, blocked_str = moderate_message(
                message=(rendered + reply)[-500:], user=user
            )
            # If the message is classified as blocked, return a CompletionData with the blocked status
            if len(blocked_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_BLOCKED,
                    reply_text=reply,
                    status_text=f"from_response:{blocked_str}",
                )
            # If the message is flagged, return a CompletionData with the flagged status
            if len(flagged_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_FLAGGED,
                    reply_text=reply,
                    status_text=f"from_response:{flagged_str}",
                )
        # If the result is OK and doesn't trigger any moderation concerns, return the CompletionData with the OK status
        return CompletionData(
            status=CompletionResult.OK, reply_text=reply, status_text=None
        )
    # Handle specific errors and exceptions that can occur during the API call
    # If the response is too long, return a CompletionData with the too long status
    except openai.error.InvalidRequestError as e:
        if "This model's maximum context length" in e.user_message:
            return CompletionData(
                status=CompletionResult.TOO_LONG, reply_text=None, status_text=str(e)
            )
        # If the request is invalid, return a CompletionData with the invalid request status
        else:
            logger.exception(e)
            return CompletionData(
                status=CompletionResult.INVALID_REQUEST,
                reply_text=None,
                status_text=str(e),
            )
    # If any other error occurs, return a CompletionData with the other error status
    except Exception as e:
        logger.exception(e)
        return CompletionData(
            status=CompletionResult.OTHER_ERROR, reply_text=None, status_text=str(e)
        )

# Define a function to process the response from the OpenAI API
async def process_response(
    user: str, thread: discord.Thread, response_data: CompletionData, is_gpt35_turbo: bool = False
):
    logger.debug(f"Started processing response for user {user} in thread {thread.name}")
    # Check if a "status" attribute exists in response_data, otherwise assume it's successful
    status = response_data.status if hasattr(response_data, "status") else "successful"

    # Extract the reply_text and status_text from the CompletionData object
    reply_text = response_data.reply_text
    status_text = response_data.status_text if hasattr(response_data, "status_text") else ""
    

    # If the status is OK, send the reply text to the user
    # If the status is flagged, send the reply text to the user and send a moderation flagged message
    if status == CompletionResult.OK or status == CompletionResult.MODERATION_FLAGGED:
        sent_message = None
        if not reply_text:
            # If is_gpt35_turbo is True, change the error message for an empty response
            if is_gpt35_turbo:
                description = f"**Invalid response** - empty response from GPT-3.5 Turbo"
            else:
                description = f"**Invalid response** - empty response"
            sent_message = await thread.send(
                embed=discord.Embed(
                    description=description,
                    color=discord.Color.yellow(),
                )
            )
        # If the reply text is too long, send a message saying the response is too long
        else:
            shorter_response = split_into_shorter_messages(reply_text)
            for r in shorter_response:
                sent_message = await thread.send(r)
        # If the status is flagged, send a moderation flagged message
        if status == CompletionResult.MODERATION_FLAGGED:
            await send_moderation_flagged_message(
                guild=thread.guild,
                user=user,
                flagged_str=status_text,
                message=reply_text,
                url=sent_message.jump_url if sent_message else "no url",
            )
            # Send a message saying the conversation has been flagged
            await thread.send(
                embed=discord.Embed(
                    description=f"⚠️ **This conversation has been flagged by moderation.**",
                    color=discord.Color.yellow(),
                )
            )
    # If the status is blocked, send a moderation blocked message and send a message saying the response has been blocked
    elif status == CompletionResult.MODERATION_BLOCKED:
        await send_moderation_blocked_message(
            guild=thread.guild,
            user=user,
            blocked_str=status_text,
            message=reply_text,
        )
        # Send a message saying the response has been blocked
        await thread.send(
            embed=discord.Embed(
                description=f"❌ **The response has been blocked by moderation.**",
                color=discord.Color.red(),
            )
        )
    # If the status is too long, close the thread
    # If the status is invalid request, send a message saying the request is invalid
    elif status is CompletionResult.TOO_LONG:
        logger.debug(f"Started processing response for user {user} in thread {thread.name}")
        await close_thread(thread)
    elif status == CompletionResult.INVALID_REQUEST:
        logger.debug(f"Status: {status}; Invalid request for user {user} in thread {thread.name}")  # NEW logging statement
        await thread.send(
            embed=discord.Embed(
                description=f"**Invalid request** - {status_text}",
                color=discord.Color.yellow(),
            )
        )
    # If the status is other error, send a message saying an error occurred
    else:
        logger.debug(f"Other error for user {user} in thread {thread.name}; status_text: {status_text}")  # NEW logging statement
        await thread.send(
            embed=discord.Embed(
                description=f"**Error** - {status_text}",
                color=discord.Color.yellow(),
            )
        )
    logger.debug(f"Finished processing response for user {user} in thread {thread.name}")  # NEW logging statement


async def generate_chat35_completion_response(messages: List[Message], user) -> CompletionData:
    inputs = [{"role": "assistant" if message.user == "Assistant" else "user", "content": message.text} for message in messages]
    
    logger.debug("Calling GPT-3.5 Turbo API with generated inputs")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=inputs
    )
    
    logger.debug("Received response from GPT-3.5 Turbo API")
    response_data = CompletionData(
        status=CompletionResult.OK,
        reply_text=response['choices'][0]['message']['content'], # Change how reply text is assigned
        status_text=None
    )

    # Log the OpenAI API response
    logger.debug(f"OpenAI API response: {response}")
    logger.debug(f"Model: {response['model']}")

    return response_data