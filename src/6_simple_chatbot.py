from typing import Annotated

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from langchain_ollama import ChatOllama


model = ChatOllama(
    model="qwen2.5-coder:7b"
)


class ChatState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]


def chatbot(state: ChatState):
    response = model.invoke(state.messages)

    return {
        "messages": [response]
    }


builder = StateGraph(ChatState)

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()


# Manual chat history
messages = []

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append(
        HumanMessage(content=user_input)
    )

    result = graph.invoke({
        "messages": messages
    })

    ai_message = result["messages"][-1]

    print("Bot:", ai_message.content)

    messages.append(ai_message)

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from langchain_ollama import ChatOllama


model = ChatOllama(
    model="qwen2.5-coder:7b"
)


class ChatState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]


def chatbot(state: ChatState):
    response = model.invoke(state.messages)

    return {
        "messages": [response]
    }


builder = StateGraph(ChatState)

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()


# Manual chat history
messages = []

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append(
        HumanMessage(content=user_input)
    )

    result = graph.invoke({
        "messages": messages
    })

    ai_message = result["messages"][-1]

    print("Bot:", ai_message.content)

    messages.append(ai_message)

print("History: ",graph)