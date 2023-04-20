# this is the file we will build the question answering system functionality into

"""Ask a question to the bot about the talentDAO database."""
import os
import asyncio
import logging
import argparse
import faiss
import pickle
import discord
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from src.completion import CompletionResult, CompletionData
from src.constants import OPENAI_API_KEY

# # Load our LangChain index created with utils.ingest.py
# # not needed right now, but in future when connected directly to a DB
# index = faiss.read_index("docs.index")
# # open our pickle file
# with open("faiss_store.pkl", "rb") as f:
#     store = pickle.load(f)

#initialize the logger
logger = logging.getLogger(__name__)

# Load the documents and components needed for the QA system
# get the parent directory of the current file
LEO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
doc_dir = LEO_DIR+r'/text'

# Load the documents and components needed for the QA system
def load_documents(directory):
    logger.info("Loading documents...")
    loader = DirectoryLoader(directory, glob='**/*.txt')
    documents = loader.load()
    logger.info("Documents loaded.")
    return documents

# Process the documents into a Chroma index
def process_documents(documents):
    logger.info("Processing documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Chroma.from_documents(texts, embeddings)
    logger.info("Documents processed.")
    return docsearch

# Load the QA chain
def load_qa_chain(llm, chain_type="stuff"):
    logger.info("Loading QA chain...")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=docsearch.as_retriever())
    logger.info("QA chain loaded.")
    return qa

# Ask a question to the bot
def ask_question(qa, question):
    logger.info(f"Asking question: {question}")
    answer = qa.run(question)
    logger.info(f"Received answer: {answer}")
    return answer

# store the documents and components needed for the QA system
documents = load_documents(doc_dir)
docsearch = process_documents(documents)
# Initialize the OpenAI instance
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
qa = load_qa_chain(llm)

### generate_qa_completion_response now uses the ask_question function
async def generate_qa_completion_response(question: str, user: str) -> CompletionData:
    logger.info("Generating QA completion response...")
    answer = await asyncio.to_thread(ask_question, qa, question)
    status_code = CompletionResult.OK
    completion_data = CompletionData(status=status_code, reply_text=answer, status_text="OK")
    logger.info("QA completion response generated.")
    return completion_data

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