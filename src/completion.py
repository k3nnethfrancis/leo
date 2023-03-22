# Import necessary libraries and modules
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

# Define constants for bot name and example conversations
MY_BOT_NAME = BOT_NAME
MY_BOT_EXAMPLE_CONVOS = EXAMPLE_CONVOS

# Define an enumeration for completion results
class CompletionResult(Enum):
    OK = 0
    TOO_LONG = 1
    INVALID_REQUEST = 2
    OTHER_ERROR = 3
    MODERATION_FLAGGED = 4
    MODERATION_BLOCKED = 5

# Define a data class for completion data, which stores the status, reply text, and status text
@dataclass
class CompletionData:
    status: CompletionResult
    reply_text: Optional[str]
    status_text: Optional[str]


# Asynchronous function to generate a response using GPT-3.5-turbo model
async def generate_completion_response(
    messages: List[Message], user: str
) -> CompletionData:
    try:
        # Create a Prompt object using the given messages and predefined bot instructions and example conversations
        prompt = Prompt(
            header=Message(
                "System", f"Instructions for {MY_BOT_NAME}: {BOT_INSTRUCTIONS}"
            ),
            examples=MY_BOT_EXAMPLE_CONVOS,
            convo=Conversation(messages + [Message(MY_BOT_NAME)]),
        )

        # Render the prompt into a string to be used as input for the GPT model
        rendered = prompt.render()

        # Generate a response using the OpenAI API with the rendered prompt
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=rendered,
            temperature=1.0,
            top_p=0.9,
            max_tokens=512,
            stop=[""],
        )

        # Extract and clean the text from the response
        reply = response.choices[0].text.strip()

        # Check if a reply was generated
        if reply:
            # Moderate the generated reply
            flagged_str, blocked_str = moderate_message(
                message=(rendered + reply)[-500:], user=user
            )

            # If the reply contains blocked content, return a moderation-blocked CompletionData
            if len(blocked_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_BLOCKED,
                    reply_text=reply,
                    status_text=f"from_response:{blocked_str}",
                )

            # If the reply contains flagged content, return a moderation-flagged CompletionData
            if len(flagged_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_FLAGGED,
                    reply_text=reply,
                    status_text=f"from_response:{flagged_str}",
                )

        # If the reply is valid, return a successful CompletionData object
        return CompletionData(
            status=CompletionResult.OK, reply_text=reply, status_text=None
        )

    # Handle InvalidRequestError exceptions from the OpenAI API
    except openai.error.InvalidRequestError as e:
        if "This model's maximum context length" in e.user_message:
            # If the input exceeds the model's maximum context length, return a too-long CompletionData
            return CompletionData(
                status=CompletionResult.TOO_LONG, reply_text=None, status_text=str(e)
            )
        else:
            # Log the exception and return an invalid-request CompletionData object
            logger.exception(e)
            return CompletionData(
                status=CompletionResult.INVALID_REQUEST,
                reply_text=None,
                status_text=str(e),
            )

    # Handle any other exceptions that might occur
    except Exception as e:
        # Log the exception and return an other-error CompletionData object
        logger.exception(e)
        return CompletionData(
            status=CompletionResult.OTHER_ERROR, reply_text=None, status_text=str(e)
        )



# Asynchronous function to process the completion response and send messages in the discord thread accordingly
async def process_response(
    user: str, thread: discord.Thread, response_data: CompletionData
):
    # Extract relevant information from the CompletionData object
    status = response_data.status
    reply_text = response_data.reply_text
    status_text = response_data.status_text

    # Handle OK and Moderation Flagged responses
    if status is CompletionResult.OK or status is CompletionResult.MODERATION_FLAGGED:
        sent_message = None

        # If the reply is empty, send an "Invalid response" message
        if not reply_text:
            sent_message = await thread.send(
                embed=discord.Embed(
                    description=f"**Invalid response** - empty response",
                    color=discord.Color.yellow(),
                )
            )
        # If the reply is not empty, split it into shorter messages and send them
        else:
            shorter_response = split_into_shorter_messages(reply_text)
            for r in shorter_response:
                sent_message = await thread.send(r)

        # If the response is flagged by moderation, send a moderation flagged message
        if status is CompletionResult.MODERATION_FLAGGED:
            await send_moderation_flagged_message(
                guild=thread.guild,
                user=user,
                flagged_str=status_text,
                message=reply_text,
                url=sent_message.jump_url if sent_message else "no url",
            )

            # Inform the user that the conversation has been flagged by moderation
            await thread.send(
                embed=discord.Embed(
                    description=f"⚠️ **This conversation has been flagged by moderation.**",
                    color=discord.Color.yellow(),
                )
            )
    # Handle Moderation Blocked responses
    elif status is CompletionResult.MODERATION_BLOCKED:
        # Send a moderation blocked message
        await send_moderation_blocked_message(
            guild=thread.guild,
            user=user,
            blocked_str=status_text,
            message=reply_text,
        )

        # Inform the user that the response has been blocked by moderation
        await thread.send(
            embed=discord.Embed(
                description=f"❌ **The response has been blocked by moderation.**",
                color=discord.Color.red(),
            )
        )
    # Handle Too Long responses
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
    # Handle any other errors
    else:
        # Inform the user that an error occurred
        await thread.send(
            embed=discord.Embed(
                description=f"**Error** - {status_text}",
                color=discord.Color.yellow(),
            )
        )