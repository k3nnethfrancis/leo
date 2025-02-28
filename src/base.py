# This code defines a set of Python classes using the dataclass decorator to represent conversation components, such as messages, conversations, and prompts. It also provides methods to render these components as formatted strings. The main classes and their functionality are described below:

# The code also imports the dataclass decorator from the dataclasses module, and the Optional and List types from the typing module. The dataclass decorator is used to automatically generate default implementations for common special methods, like __init__, __repr__, and others, based on class annotations.

import os
from dataclasses import dataclass
from typing import Optional, List

# The SEPARATOR_TOKEN variable is an empty string, used to separate different components when rendering strings in the Conversation and Prompt classes
SEPARATOR_TOKEN = "<|endoftext|>"

 # Message: Represents an individual message in a conversation. Each message has a user (sender) and an optional text. The render method returns the message formatted as a string, with the user and text separated by a colon.
@dataclass(frozen=True)
class Message:
    user: str
    text: Optional[str] = None

    def render(self):
        result = self.user + ":"
        if self.text is not None:
            result += " " + self.text
        return result

# Conversation: Represents a conversation with a list of Message objects. The prepend method adds a new message to the beginning of the conversation. The render method returns the conversation formatted as a string, with messages separated by a new line and the SEPARATOR_TOKEN.
@dataclass
class Conversation:
    messages: List[Message]

    def prepend(self, message: Message):
        self.messages.insert(0, message)
        return self

    def render(self):
        return f"\n{SEPARATOR_TOKEN}".join(
            [message.render() for message in self.messages]
        )

# Config: Represents a configuration for a conversation, including the name, instructions, and a list of example conversations.
@dataclass(frozen=True)
class Config:
    name: str
    instructions: str
    example_conversations: List[Conversation]


# Prompt: Represents a prompt for a conversation, with a header message, a list of example conversations, and the current conversation. The render method returns the prompt formatted as a string, including the header, example conversations, and the current conversation, all separated by a new line and the SEPARATOR_TOKEN
@dataclass(frozen=True)
class Prompt:
    header: Message
    examples: List[Conversation]
    convo: Conversation

    def render(self):
        return f"\n{SEPARATOR_TOKEN}".join(
            [self.header.render()]
            + [Message("System", "Example conversations:").render()]
            + [conversation.render() for conversation in self.examples]
            + [Message("System", "Current conversation:").render()]
            + [self.convo.render()],
        )


# building the BaseRetriever class for docuemnt search (QA). /ask command

from abc import ABC, abstractmethod
from src.constants import OPENAI_API_KEY
from langchain import OpenAI
from langchain.schema import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import DirectoryLoader, TextLoader
# get the parent directory of the current file
LEO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Initialize the OpenAI instance
llm = OpenAI(openai_api_key=OPENAI_API_KEY)

## OG implementation with chroma vectorstore
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
    



# LIBS for FAISS and Testing multi-retrieval QA chains #
# from langchain.chains.router import MultiRetrievalQAChain
# from langchain.llms import OpenAI
# # faiss
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.document_loaders import TextLoader
# from langchain.vectorstores import FAISS

# from langchain.embeddings import OpenAIEmbeddings
# from langchain.document_loaders import TextLoader
# from langchain.vectorstores import FAISS

# Testing multi-retrieval QA chains
# class BaseRetriever(ABC):
#     def __init__(self):
#         self.qa_path = LEO_DIR + r'/text/'
#         self.onboard_path = LEO_DIR + r'/text/projectText/'
#         self.qa_docs = TextLoader(self.qa_path + 'talentDAOcore.txt').load_and_split()
#         self.qa_retriever = FAISS.from_documents(self.qa_docs, OpenAIEmbeddings()).as_retriever()
#         self.onboard_docs = TextLoader(self.onboard_path + 'talentDAOProjects.txt').load_and_split()
#         self.onboard_retriever = FAISS.from_documents(self.onboard_docs, OpenAIEmbeddings()).as_retriever()
#         self.llm = OpenAI(openai_api_key=OPENAI_API_KEY)
#         self.retriever_infos = [
#         {
#             "name": "talentDAO core team info", 
#             "description": "Good for answering questions about talentDAO's core team members", 
#             "retriever": self.qa_retriever
#         },
#         {
#             "name": "talentDAO project info", 
#             "description": "Good for responding to intros with relevant projects", 
#             "retriever": self.onboard_retriever}]
        
#     @abstractmethod
#     def search(self, query: str) -> List[Document]:
#         """Responds to a query about the users documents.
#             Args:
#                 query: string to find relevant docs for
#             Returns:
#                 response to query
#         """
#         chain = MultiRetrievalQAChain.from_retrievers(self.llm, self.retriever_infos, retriever_descriptions=self.retriever_infos verbose=True)
#         result = chain.run(query)
#         return result.split('\n')
#     def get_relevant_projects(self, intro: str) -> List[str]:
#         """Responds to a query about the users documents.
#             Args:
#                 query: string to find relevant docs for
#             Returns:
#                 response to query
#         """
#         query = f"What are one or two projects that might be interesting for a user with the following intro: {intro}? \nPlease respond in a helpful, welcoming tone."
#         chain = MultiRetrievalQAChain.from_retrievers(self.llm, self.retriever_infos, verbose=True)
#         result = chain.run(query)
#         return result.split('\n') # Split the response text into a list of sentences