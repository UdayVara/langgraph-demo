from typing import Literal, List

from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

model = ChatOllama(
    model="qwen2.5-coder:7b"
)

strParser = StrOutputParser()


class PostWorkflowState(BaseModel):
    topic:str = Field(title="Topic", description="Topic of this post")
    content:str = Field(title="Content", description="Content of this post")
    status:Literal["Approved","Needs Improvement"] = Field(..., title="Status", description="Status of this post")
    feedback:str = Field(title="Feedback", description="Feedback of this post after reviewing")
    iteration:int = Field(title="Iteration", description="Iteration of this post")
    post_history:List[str]= Field(title="Post History", description="Post history of this post")
    feedback_history:List[str]= Field(title="Post History", description="Post history of this post"),
    max_iteration:int = Field(title="Max Iteration", description="Max iteration of this post")


def generatePost(state:PostWorkflowState):
    topic = state.topic
    generatePostPrompt = PromptTemplate(
        template="""
    You are an expert social media copywriter.

    Generate an engaging Twitter/X post based on the following topic.

    Topic:
    {topic}

    Instructions:
    - Create a single Twitter/X post.
    - Keep it under 280 characters.
    - Make it engaging, natural, and conversational.
    - Focus on the main idea of the topic.
    - If feedback is provided, incorporate it into the post.
    - Do not use excessive emojis.
    - Include relevant hashtags only when they add value (maximum 3).
    - Return only the final tweet text.

    Twitter Post:
    """,
    input_variables=["topic"],
    )


    chain = generatePostPrompt | model | strParser


    result = chain.invoke({
        "topic": topic,
    })

    return {
        "content": result,
    }

class EvaluatePostSchema(BaseModel):
    result:Literal["Approved","Needs Improvement"] = Field(title="Result", description="Result of this post")
    feedback:str = Field(title="Feedback", description="Feedback of this post after reviewing")

evaulatePostParser = PydanticOutputParser(
    pydantic_object=EvaluatePostSchema
)

def evaluate_post(state:PostWorkflowState):

    topic = state.topic
    feedback = state.feedback
    post = state.content

    evaluatePostPrompt = PromptTemplate(
        template="""
    You are an expert social media content reviewer.

    Review the following Twitter/X post and determine whether it is ready to publish.

    Topic:
    {topic}

    User Feedback (if any):
    {feedback}

    Twitter Post:
    {post}

    Evaluation Criteria:
    1. The post is relevant to the topic.
    2. The message is clear and easy to understand.
    3. The content is engaging and suitable for Twitter/X.
    4. The post is concise and not repetitive.
    5. If user feedback is provided, the post properly incorporates it.
    6. The post does not contain misleading, offensive, or irrelevant content.

    Instructions:
    - If the post satisfies all criteria, set result to "Approved".
    - If the post fails any criteria, set result to "Needs Improvement".
    - Always provide actionable feedback.
    - If approved, feedback should briefly explain why the post is good.
    - If improvements are needed, feedback should clearly explain what should be changed.

    {format_instructions}
    """,
        input_variables=["topic", "feedback", "post"],
        partial_variables={
            "format_instructions": evaulatePostParser.get_format_instructions()
        }
    )


    evaulateChain = evaluatePostPrompt | model | evaulatePostParser

    result = evaulateChain.invoke({
        "topic": topic,
        "feedback": feedback,
        "post": post,
    })


    return {
        'feedback':result.feedback,
        'status': result.result,
    }


def optimize_post(state: PostWorkflowState):
    post = state.content
    feedback = state.feedback

    optimize_prompt = PromptTemplate(
        template="""
You are an expert social media copywriter.

Your task is to improve a Twitter/X post based on reviewer feedback.

Current Post:
{post}

Reviewer Feedback:
{feedback}

Requirements:
- Fix all issues mentioned in the feedback.
- Preserve the original intent.
- Improve clarity and engagement.
- Improve readability.
- Keep the post under 280 characters.
- Return only the improved post.
- Do not explain your changes.

Improved Post:
""",
        input_variables=["post", "feedback"],
    )

    chain = optimize_prompt | model | StrOutputParser()

    optimized_post = chain.invoke({
        "post": post,
        "feedback": feedback,
    })

    return {
        "content": optimized_post,
        "iteration": state.iteration + 1,
        "post_history": state.post_history + [post],
        "feedback_history": state.feedback_history + [feedback],
    }

def router_evaulator(state:PostWorkflowState):
    iteration = state.iteration
    status = state.status
    max_iteration = state.max_iteration

    # Force optimize
    if iteration == 0:
        return "needs_improvement"

    if status == "Approved" or iteration >= max_iteration:
        return "approved"
    else:
        return "needs_improvement"

graph = StateGraph(
    state_schema=PostWorkflowState,
)


graph.add_node("generate",generatePost)
graph.add_node("evaluate",evaluate_post)
graph.add_node("optimize",optimize_post)


# Edges
graph.add_edge(START,"generate")
graph.add_edge("generate","evaluate")

graph.add_conditional_edges("evaluate",router_evaulator,{
    "approved":END,
    "needs_improvement":"optimize",
})

graph.add_edge("optimize","evaluate")

workflow = graph.compile()

initial_state = {
    "topic": "What if pigeons secretly managed the global banking system?",
    "content": "",
    "status": "Needs Improvement",
    "feedback": "",
    "iteration": 0,
    "post_history": [],
    "feedback_history": [],
    "max_iteration": 3,
}

result = workflow.invoke(initial_state)

print(result)


