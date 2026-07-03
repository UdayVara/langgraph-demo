from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel

model = ChatOllama(
    model="qwen2.5-coder:7b"
)

memory = MemorySaver()


class ChatState(BaseModel):
    messages: Annotated[list, add_messages]


def chatbot(state: ChatState):

    response = model.invoke(
        state.messages
    )

    return {
        "messages": [response]
    }


builder = StateGraph(ChatState)

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(
    checkpointer=memory
)


def chat(thread_id: str, query: str):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=query)
            ]
        },
        config=config
    )

    return result["messages"][-1].content


def stream_chat(thread_id: str, query: str):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    messages = graph.get_state(config).values.get(
        "messages",
        []
    )

    messages.append(
        HumanMessage(content=query)
    )

    stream = model.stream(messages)

    full_response = ""

    for chunk in stream:

        text = chunk.content

        if text:
            full_response += text
            yield text

    graph.invoke(
        {
            "messages": [
                HumanMessage(content=query)
            ]
        },
        config=config
    )