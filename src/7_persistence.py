from typing import Annotated

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

# --------------------
# LLM
# --------------------

model = ChatOllama(
    model="qwen2.5-coder:7b"
)

parser = StrOutputParser()


# --------------------
# State
# --------------------

class GraphSchema(BaseModel):
    topic: str = Annotated[str, "Topic of joke"]
    joke: str = Annotated[str, "Generated joke"]
    explanation: str = Annotated[str, "Joke explanation"]


# --------------------
# Nodes
# --------------------

def generate_joke(state: GraphSchema):
    prompt = PromptTemplate(
        template="""
Generate a funny one-line joke about: {topic}
""",
        input_variables=["topic"]
    )

    chain = prompt | model | parser

    joke = chain.invoke({
        "topic": state.topic
    })

    print("\nGENERATED JOKE")
    print(joke)

    return {
        "joke": joke
    }


def generate_explanation(state: GraphSchema):
    prompt = PromptTemplate(
        template="""
Explain the following joke.

Joke:
{joke}

Topic:
{topic}
""",
        input_variables=["joke", "topic"]
    )

    chain = prompt | model | parser

    explanation = chain.invoke({
        "joke": state.joke,
        "topic": state.topic
    })

    print("\nGENERATED EXPLANATION")
    print(explanation)

    return {
        "explanation": explanation
    }


# --------------------
# Graph
# --------------------

graph = StateGraph(GraphSchema)

graph.add_node("generate_joke", generate_joke)
graph.add_node("generate_explanation", generate_explanation)

graph.add_edge(START, "generate_joke")
graph.add_edge("generate_joke", "generate_explanation")
graph.add_edge("generate_explanation", END)


# --------------------
# Checkpointer
# --------------------

memory = MemorySaver()

workflow = graph.compile(
    checkpointer=memory
)


# --------------------
# Thread Configs
# --------------------

config_pasta = {
    "configurable": {
        "thread_id": "pasta-thread"
    }
}

config_pizza = {
    "configurable": {
        "thread_id": "pizza-thread"
    }
}


# --------------------
# Run Pasta Thread
# --------------------

print("\n====================")
print("RUNNING PASTA THREAD")
print("====================")

workflow.invoke(
    {
        "topic": "Pasta",
        "joke": "",
        "explanation": ""
    },
    config=config_pasta
)


# --------------------
# Run Pizza Thread
# --------------------

print("\n====================")
print("RUNNING PIZZA THREAD")
print("====================")

workflow.invoke(
    {
        "topic": "Pizza",
        "joke": "",
        "explanation": ""
    },
    config=config_pizza
)


# --------------------
# Retrieve Latest State
# --------------------

print("\n====================")
print("LATEST PASTA STATE")
print("====================")

pasta_state = workflow.get_state(config_pasta)

print(pasta_state.values)


print("\n====================")
print("LATEST PIZZA STATE")
print("====================")

pizza_state = workflow.get_state(config_pizza)

print(pizza_state.values)


# --------------------
# Retrieve History
# --------------------

print("\n====================")
print("PASTA STATE HISTORY")
print("====================")

for i, checkpoint in enumerate(
    workflow.get_state_history(config_pasta),
    start=1
):
    print(f"\nCheckpoint {i}")
    print(checkpoint.values)


# --------------------
# Demonstrate Retrieval Later
# --------------------

print("\n====================")
print("RETRIEVE PASTA AGAIN")
print("====================")

retrieved = workflow.get_state({
    "configurable": {
        "thread_id": "pasta-thread"
    }
})

print("Topic:", retrieved.values["topic"])
print("Joke:", retrieved.values["joke"])
print("Explanation:", retrieved.values["explanation"])