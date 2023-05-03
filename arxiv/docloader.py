import os
import logging
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from src.constants import OPENAI_API_KEY

# Initialize the logger
logger = logging.getLogger(__name__)

# Get the parent directory of the current file
LEO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def load_documents(directory):
    logger.info("Loading documents...")
    loader = DirectoryLoader(directory, glob='**/*.txt')
    documents = loader.load()
    logger.info("Documents loaded.")
    return documents

def process_documents(documents):
    logger.info("Processing documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Chroma.from_documents(texts, embeddings)
    logger.info("Documents processed.")
    return docsearch

def get_loaded_documents():
    # Load and process the documents
    documents = load_documents(document_path)
    return documents

def get_docsearch_instance(documents):
    # Process documents
    docsearch = process_documents(documents)
    return docsearch

def get_llm_instance():
    # Initialize the OpenAI instance
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    return llm

# Define the path where the documents are located
document_path = LEO_DIR + r'/text'

# # Load and process the documents
# documents = load_documents(document_path)
# docsearch = process_documents(documents)

# # Initialize the OpenAI instance
# llm = OpenAI(openai_api_key=OPENAI_API_KEY)