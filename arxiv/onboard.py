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
from src.constants import OPENAI_API_KEY


# get the parent directory of the current file
LEO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

examples_df = pd.read_csv(LEO_DIR + r'/data/csv/intro_examples.csv')
examples_dict = examples_df.to_dict('records')

def get_dynamic_prompt() -> FewShotPromptTemplate:
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

def is_intro(model, message: str) -> bool:
    prompt = get_dynamic_prompt().format(input=message)
    response = model(prompt)
    classification_result = response.strip()
    return classification_result.lower() == "true"

# retrieve context and recommend projects

class BaseRetriever(ABC):
    def __init__(self):
        self.loader = DirectoryLoader('../', glob="**/*.txt", loader_cls=TextLoader)
        self.index = VectorstoreIndexCreator().from_loaders([self.loader])
    @abstractmethod
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get texts relevant for a query.

        Args:
            query: string to find relevant tests for

        Returns:
            List of relevant documents
        """
# path = LEO_DIR + r'/text/projectText/'
# loader = DirectoryLoader(path, loader_cls=TextLoader)
# index = VectorstoreIndexCreator().from_loaders([loader])

def intro2query(intro: str) -> str:
    return f"What are one or two projects that might be interesting for a user with the following intro: {intro}? \nPlease exlpained in a helpful tone."

class ProjectRecommender:
    def __init__(self):
        path = LEO_DIR + r'/text/projectText/'
        self.loader = DirectoryLoader(path, loader_cls=TextLoader)
        self.index = VectorstoreIndexCreator().from_loaders([self.loader])

    def get_relevant_projects(self, intro: str) -> List[str]:
        query = intro2query(intro)
        # Split the response text into a list of sentences
        result = self.index.query(query)
        return result.split('\n')