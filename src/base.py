# This code defines a set of Python classes using the dataclass decorator to represent conversation components, such as messages, conversations, and prompts. It also provides methods to render these components as formatted strings. The main classes and their functionality are described below:

# The code also imports the dataclass decorator from the dataclasses module, and the Optional and List types from the typing module. The dataclass decorator is used to automatically generate default implementations for common special methods, like __init__, __repr__, and others, based on class annotations.

            
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