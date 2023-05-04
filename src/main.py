import os
import asyncio
import logging
import openai
import discord
from discord import Message as DiscordMessage
from typing import List, Tuple
from langchain import OpenAI

from src.base import (
    Message,
    Conversation
)
from src.constants import (
    BOT_INVITE_URL,
    DISCORD_BOT_TOKEN,
    EXAMPLE_CONVOS,
    ACTIVATE_THREAD_PREFX,
    MAX_THREAD_MESSAGES,
    SECONDS_DELAY_RECEIVING_MSG,
    OPENAI_API_KEY,
    TARGET_CHANNEL_ID
)
from src.utils import (
    logger,
    should_block,
    close_thread,
    is_last_message_stale,
    discord_message_to_message,
    save_messages_to_file
)
from src.qa import (
    generate_qa_completion_response,
    process_qa_response
)
from src import completion
from src.completion import (
    #generate_completion_response,
    process_response,
    generate_chat_completion_response
)
from src.moderation import (
    moderate_message,
    send_moderation_blocked_message,
    send_moderation_flagged_message,
)
from src.onboard import (
    OnboardPromptTemplate,
    IntroDetector,
    generate_onboard_completion_response,
    process_onboard_response,
)

# Set up logging
# logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG
# logging.basicConfig(level=logging.INFO)  # Set logging level to INFO
logger = logging.getLogger("leo_logger")
logger.setLevel(logging.INFO)

# Add a StreamHandler to output messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


# Initialize the langchain OpenAI instance
# llm = OpenAI(openai_api_key=OPENAI_API_KEY)

# Set the intents to include the message content
intents = discord.Intents.default()
intents.message_content = True

# Instantiate a discord.Client object with the specified intents
client = discord.Client(intents=intents)

# Instantiate a CommandTree object that will hold the bot's command hierarchy
tree = discord.app_commands.CommandTree(client)

# Event triggered when the bot starts and logs in
@client.event
async def on_ready():
    # Log bot's username and invite URL
    logger.info(f"We have logged in as {client.user}. Invite URL: {BOT_INVITE_URL}")
    # Set the bot's name and initialize an empty list to store example conversations
    completion.MY_BOT_NAME = client.user.name
    completion.MY_BOT_EXAMPLE_CONVOS = []
    # Iterate through the EXAMPLE_CONVOS list and create corresponding Conversation objects
    for c in EXAMPLE_CONVOS:
        messages = []
        for m in c.messages:
            #if m.user == "Lenard":
            if m.user == "leo-bot":
                messages.append(Message(user=client.user.name, text=m.text))
            else:
                messages.append(m)
        completion.MY_BOT_EXAMPLE_CONVOS.append(Conversation(messages=messages))
    # Sync the CommandTree with the bot's commands
    await tree.sync()

## Chat w/ GPT-4 / GPT35turbp##
@tree.command(name="chat", description="Create a new thread for conversation with GPT-4 (whatever is set in completions.py)")
@discord.app_commands.checks.has_permissions(send_messages=True)
@discord.app_commands.checks.has_permissions(view_channel=True)
@discord.app_commands.checks.bot_has_permissions(send_messages=True)
@discord.app_commands.checks.bot_has_permissions(view_channel=True)
@discord.app_commands.checks.bot_has_permissions(manage_threads=True)
async def chat_command(int: discord.Interaction, message: str):
    try:
        # only support creating thread in text channel
        if not isinstance(int.channel, discord.TextChannel):
            return

        # block servers not in allow list
        if should_block(guild=int.guild):
            return
        
        # Get the user who issued the chat command
        user = int.user
        logger.info(f"Chat command by {user} {message[:20]}")
        try:
            # moderate the message
            flagged_str, blocked_str = moderate_message(message=message, user=user)
            await send_moderation_blocked_message(
                guild=int.guild,
                user=user,
                blocked_str=blocked_str,
                message=message,
            )
            # If the message is blocked by moderation, notify the user and return
            if len(blocked_str) > 0:
                # message was blocked
                await int.response.send_message(
                    f"Your prompt has been blocked by moderation.\n{message}",
                    ephemeral=True,
                )
                return
            
            # Create an embed for the chat command and add user's name and message
            embed = discord.Embed(
                description=f"<@{user.id}> wants to chat! 🤖💬",
                color=discord.Color.green(),
            )
            embed.add_field(name=user.name, value=message)

            # If the message is flagged by moderation, add a warning to the embed
            if len(flagged_str) > 0:
                # message was flagged
                embed.color = discord.Color.yellow()
                embed.title = "⚠️ This prompt was flagged by moderation."
            
            # Send the embed as a response
            await int.response.send_message(embed=embed)
            response = await int.original_response()

            # Send a notification if the message was flagged by moderation
            await send_moderation_flagged_message(
                guild=int.guild,
                user=user,
                flagged_str=flagged_str,
                message=message,
                url=response.jump_url,
            )
        except Exception as e:
            logger.exception(e)
            await int.response.send_message(
                f"Failed to start chat {str(e)}", ephemeral=True
            )
            return

        # create the thread for the conversation
        thread = await response.create_thread(
            name=f"{ACTIVATE_THREAD_PREFX} {user.name[:20]} - {message[:30]}",
            slowmode_delay=1,
            reason="gpt-bot",
            auto_archive_duration=60,
        )
        # Show the bot is typing in the thread
        async with thread.typing():
            logger.debug("Generating response using GPT-4")
            # fetch completion
            messages = [Message(user=user.name, text=message)]
            response_data = await generate_chat_completion_response(messages=messages, user=user)
            logger.debug("Response generated by GPT-4")
            # send the result
            await process_response(
                user=user, thread=thread, response_data=response_data
            )
    except Exception as e:
        logger.exception(e)
        await int.edit_original_response(content="An error occurred while using GPT-3.5 Turbo/GPT-4: {}".format(e))

## ASK ##
@tree.command(name="ask", description="Ask a question to the bot.")
async def ask_command(int: discord.Interaction, question: str):
    try:
        # Send an initial "thinking" response
        await int.response.send_message("🤖 thinking...", ephemeral=False)

        # Get the user who issued the ask command
        user = int.user
        logger.info(f"Ask command by {user} {question}")

        # Fetch the QA response
        #response_data = await generate_qa_completion_response(question=question, user=user)
        response_data = await generate_qa_completion_response(query=[Message(user=str(user), text=str(question))], user=user)

        # Process and send the response
        await process_qa_response(user=user, interaction=int, question=question, response_data=response_data)
       
    except openai.error.RateLimitError as rle:  # Import openai at the beginning of the file if not already done.
        logger.exception(rle)
        error_message = "The bot is currently rate-limited. Please wait a moment and try again."
        await int.followup.send(content=error_message, ephemeral=True)

    except Exception as e:
        logger.exception(e)
        try:
            # Edit the original response message with an error message
            await int.edit_original_response(content=f"Failed to answer question. {str(e)}")
        except Exception as e2:
            logger.exception(e2)
            await int.followup.send(content=f"Failed to answer question. {str(e)}", ephemeral=True)

## ONBOARD ##
@tree.command(name="onboard", description="Read intro messages from target channel and recommend projects to users")
async def onboard_users_command(int: discord.Interaction):
    # Defer the response to prevent the interaction from expiring
    await int.response.defer(ephemeral=True)

    ### Message logging ###
    async def fetch_and_save_messages(client: discord.Client, limit: int = 10) -> List[Tuple[str, str, int]]:
        # Fetch the last 100 messages from the desired channel
        channel = await client.fetch_channel(TARGET_CHANNEL_ID)
        
        messages = []
        # Iterate through the channel history and add the messages to the list
        async for message in channel.history(limit=limit):
            messages.append((message.content, message.author.name, message.id))

        # Save the messages to a file in the msg_log folder with the file name as the channel ID
        save_messages_to_file(messages, folder="msg_log", filename=f"{TARGET_CHANNEL_ID}")

        return messages
    #create intro detector object
    intro_detector = IntroDetector()
    try:
        # Fetch the last `limit` messages
        messages = await fetch_and_save_messages(client, limit=10)
        
        target_channel = await client.fetch_channel(TARGET_CHANNEL_ID)
    
        for content, author_name, message_id in messages:
            if intro_detector.is_intro(message=content):
                # Get the message object using the message ID
                original_message = await target_channel.fetch_message(message_id)

                # Get the user object (author) from the message object
                author = original_message.author

                # Get recommended projects
                recommended_projects = await generate_onboard_completion_response(intro=content, user=author_name)

                # Process and send the response
                await process_onboard_response(user=author, interaction=int, message_id=message_id, response_data=recommended_projects)

                # # Find the original message using the message ID
                # original_message = await target_channel.fetch_message(message_id)
    
                # # Send a reply to the original message ONLY if it's an intro
                # await original_message.reply(recommended_projects)
        
        # Edit the original deferred response
        await int.edit_original_response(content=f"Processed {len(messages)} recent messages for onboarding.")

    except Exception as e:
        logger.exception("Error in onboard_users_command: %s", e)
        await int.edit_original_response(content=f"Failed to process messages for onboarding. {str(e)}")




#### THREAD HANDLING ####
# calls for each message
# Event that triggers when a message is sent in a channel or thread
@client.event
async def on_message(message: DiscordMessage):
    try:
        # block servers not in allow list
        if should_block(guild=message.guild):
            return

        # ignore messages from the bot
        if message.author == client.user:
            return

        # ignore messages not in a thread
        channel = message.channel
        if not isinstance(channel, discord.Thread):
            return

        # ignore threads not created by the bot
        thread = channel
        if thread.owner_id != client.user.id:
            return

        # ignore threads that are archived locked or title is not what we want
        if (
            thread.archived
            or thread.locked
            or not thread.name.startswith(ACTIVATE_THREAD_PREFX)
        ):
            # ignore this thread
            return
        
        # Close the thread if the message count exceeds the maximum limit
        if thread.message_count > MAX_THREAD_MESSAGES:
            # too many messages, no longer going to reply
            await close_thread(thread=thread)
            return

        # moderate the message
        flagged_str, blocked_str = moderate_message(
            message=message.content, user=message.author
        )
        await send_moderation_blocked_message(
            guild=message.guild,
            user=message.author,
            blocked_str=blocked_str,
            message=message.content,
        )
        # If the message is blocked by moderation, delete it and notify the thread
        if len(blocked_str) > 0:
            try:
                await message.delete()
                await thread.send(
                    embed=discord.Embed(
                        description=f"❌ **{message.author}'s message has been deleted by moderation.**",
                        color=discord.Color.red(),
                    )
                )
                return
            except Exception as e:
                await thread.send(
                    embed=discord.Embed(
                        description=f"❌ **{message.author}'s message has been blocked by moderation but could not be deleted. Missing Manage Messages permission in this Channel.**",
                        color=discord.Color.red(),
                    )
                )
                return
        # Inform the thread if the message was flagged by moderation
        await send_moderation_flagged_message(
            guild=message.guild,
            user=message.author,
            flagged_str=flagged_str,
            message=message.content,
            url=message.jump_url,
        )
        if len(flagged_str) > 0:
            await thread.send(
                embed=discord.Embed(
                    description=f"⚠️ **{message.author}'s message has been flagged by moderation.**",
                    color=discord.Color.yellow(),
                )
            )

        # wait a bit in case user has more messages
        if SECONDS_DELAY_RECEIVING_MSG > 0:
            await asyncio.sleep(SECONDS_DELAY_RECEIVING_MSG)
            if is_last_message_stale(
                interaction_message=message,
                last_message=thread.last_message,
                bot_id=client.user.id,
            ):
                # there is another message, so ignore this one
                return

        logger.info(
            f"Thread message to process - {message.author}: {message.content[:50]} - {thread.name} {thread.jump_url}"
        )
        
        # Fetch messages from the thread, apply relevant conversions, and reverse the order
        channel_messages = [
            discord_message_to_message(message)
            async for message in thread.history(limit=MAX_THREAD_MESSAGES)
        ]
        channel_messages = [x for x in channel_messages if x is not None]
        channel_messages.reverse()

        # generate the response
        async with thread.typing():
            response_data = await generate_chat_completion_response(
                messages=channel_messages, user=message.author
            )

        if is_last_message_stale(
            interaction_message=message,
            last_message=thread.last_message,
            bot_id=client.user.id,
        ):
            # there is another message and its not from us, so ignore this response
            return

        # send response
        await process_response(
            user=message.author, thread=thread, response_data=response_data
        )
    except Exception as e:
        logger.exception(e)

client.run(DISCORD_BOT_TOKEN)