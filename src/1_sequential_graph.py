from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

model = ChatOllama(
    model="qwen2.5-coder:7b"
)

parser = StrOutputParser()


class GraphSchema(BaseModel):
    topic: str = Field(..., description="The Topic of the graph")
    outline: str = Field(..., description="The outline of the graph")
    blog: str = Field(..., description="The Blog of the graph")


def generate_outline(state: GraphSchema) -> GraphSchema:
    topic = state.topic

    prompt = PromptTemplate(template='''
        Generate a detailed SEO-friendly blog outline for the topic: "{topic}".

Requirements:
- Create an engaging title
- Write a short summary
- Add introduction, main sections, sub-sections, FAQs, and conclusion
- Keep the structure logical and easy to read
- Include important SEO keywords naturally
    ''', input_variables=["topic"])

    outline_chain = prompt | model | parser

    state.outline = outline_chain.invoke({
        "topic": topic,
    })

    return state


def generate_blog(state: GraphSchema) -> GraphSchema:
    outline = state.outline

    prompt = PromptTemplate(template='''
        Generate a complete blog article using this outline: "{outline}".

Requirements:
- Follow the outline strictly
- Use markdown formatting
- Write clear, engaging, human-like content
- Add proper headings and subheadings
- Keep paragraphs short and readable
- Include examples and practical insights where relevant
- Add FAQs and conclusion
- Return only the final blog article
    ''',
                            input_variables=["outline"])

    blog_chain = prompt | model | parser

    state.blog = blog_chain.invoke({
        "outline": outline,
    })

    return state


graph = StateGraph(
    state_schema=GraphSchema
)

# Nodes
graph.add_node("create_outline", generate_outline)
graph.add_node("generate_blog", generate_blog)

# Edges
graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "generate_blog")
graph.add_edge("generate_blog", END)

workflow = graph.compile()

topic = input("Topic: ")

result = workflow.invoke({'topic': topic,'outline':"",'blog':''})

print("OUTLINE ------------------------------------------------- \n",result["outline"])
print("BLOG --------------------------------------------------- \n",result["blog"])

