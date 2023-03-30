# Description: This file contains the code for generating a response from the OpenAI API
# Import necessary libraries and modules: enum, dataclass, openai, various functions, and constants from other files.
from enum import Enum
from dataclasses import dataclass
import openai
from src.moderation import moderate_message
from typing import Optional, List
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


# Asynchronous function to generate a response using GPT-3.5-turbo model
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
            engine="gpt-3.5-turbo",
            prompt=rendered,
            temperature=1.0,
            top_p=0.9,
            max_tokens=512,
            stop=[""],
        )
        # Extract and clean the reply text from the generated response
        reply = response.choices[0].text.strip()

        # Check if a reply was generated
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
            # If the input exceeds the model's maximum context length, return a too-long CompletionData
            return CompletionData(
                status=CompletionResult.TOO_LONG, reply_text=None, status_text=str(e)
            )
        # If the request is invalid, return a CompletionData with the invalid request status
        else:
            # Log the exception and return an invalid-request CompletionData object
            logger.exception(e)
            return CompletionData(
                status=CompletionResult.INVALID_REQUEST,
                reply_text=None,
                status_text=str(e),
            )
    # If any other error occurs, return a CompletionData with the other error status
    except Exception as e:
        # Log the exception and return an other-error CompletionData object
        logger.exception(e)
        return CompletionData(
            status=CompletionResult.OTHER_ERROR, reply_text=None, status_text=str(e)
        )

# Define a function to process the response from the OpenAI API
async def process_response(
    user: str, thread: discord.Thread, response_data: CompletionData
):
    '''
    Process the response from the OpenAI API and send the response to the user.
    
    Parameters:
        user (str): The user's Discord ID.
        thread (discord.Thread): The thread the user is in.
        response_data (CompletionData): The data returned from the OpenAI API.
        
        Returns:
            None
    '''
    # Extract the status, reply text, and status text from the CompletionData object
    status = response_data.status
    reply_text = response_data.reply_text
    status_text = response_data.status_text
    # If the status is OK, send the reply text to the user
    # If the status is flagged, send the reply text to the user and send a moderation flagged message
    if status is CompletionResult.OK or status is CompletionResult.MODERATION_FLAGGED:
        sent_message = None
        # If the reply text is empty, send an empty response message
        if not reply_text:
            sent_message = await thread.send(
                embed=discord.Embed(
                    description=f"**Invalid response** - empty response",
                    color=discord.Color.yellow(),
                )
            )
        # If the reply text is too long, send a message saying the response is too long
        else:
            shorter_response = split_into_shorter_messages(reply_text)
            for r in shorter_response:
                sent_message = await thread.send(r)
        # If the status is flagged, send a moderation flagged message
        if status is CompletionResult.MODERATION_FLAGGED:
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
    elif status is CompletionResult.MODERATION_BLOCKED:
        # Send a moderation blocked message
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
        # Close the thread due to the response being too long
        await close_thread(thread)
    # Handle Invalid Request responses
    elif status is CompletionResult.INVALID_REQUEST:
        # Inform the user that the request was invalid
        await thread.send(
            embed=discord.Embed(
                description=f"**Invalid request** - {status_text}",
                color=discord.Color.yellow(),
            )
        )
    # If the status is other error, send a message saying an error occurred
    else:
        # Inform the user that an error occurred
        await thread.send(
            embed=discord.Embed(
                description=f"**Error** - {status_text}",
                color=discord.Color.yellow(),
            )
        )