# this is the file we will build the question answering system functionality into

"""Ask a question to the bot about the talentDAO database."""
import discord
import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
import argparse
from src.completion import CompletionResult, CompletionData
from src.base import Message
from src.completion import generate_completion_response

# Load our LangChain index
index = faiss.read_index("docs.index")
# open our pickle file
with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

# CLI functionality for testing
# parser = argparse.ArgumentParser(description='Ask a question to the bot.')
# parser.add_argument('question', type=str, help='The question to ask the bot.')
# args = parser.parse_args()
# store.index = index
# chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
# result = chain({"question": args.question})
# print(f"Answer: {result['answer']}")
# print(f"Sources: {result['sources']}")

async def generate_qa_completion_response(question: str, user: str) -> CompletionData:
    messages = [Message(user=user.name, text=question)]
    return await generate_completion_response(messages=messages, user=user)

async def process_qa_response(user: str, interaction: discord.Interaction, response_data: CompletionData):
    status = response_data.status
    reply_text = response_data.reply_text
    status_text = response_data.status_text

    if status is CompletionResult.OK or status is CompletionResult.MODERATION_FLAGGED:
        if reply_text:
            await interaction.response.send_message(reply_text)
        else:
            await interaction.response.send_message("No response generated.", ephemeral=True)

        if status is CompletionResult.MODERATION_FLAGGED:
            await interaction.channel.send(
                f"⚠️ **This conversation has been flagged by moderation.**"
            )
    elif status is CompletionResult.MODERATION_BLOCKED:
        await interaction.channel.send(
            f"❌ **The response has been blocked by moderation.**"
        )

    elif status is CompletionResult.TOO_LONG or status is CompletionResult.INVALID_REQUEST or status is CompletionResult.OTHER_ERROR:
        await interaction.response.send_message(
            f"**Error** - {status_text}", ephemeral=True
        )