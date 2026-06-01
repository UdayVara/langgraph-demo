from typing import Literal, Dict

from langchain_core.output_parsers import PydanticOutputParser, format_instructions, StrOutputParser
from langchain_core.prompts import PromptTemplate, prompt
from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

model = ChatOllama(
    model="qwen2.5-coder:7b"
)

class FeedbackSchemaState(BaseModel):
    user_feedback:str = Field(title="User Feedback")
    feedback_type:Literal["positive","negative",""] = Field(title="Positive or Negative Feedback")
    diagnosis:dict = Field(description="Diagnosis Feedback for negative and positive feedback ")
    response:str = Field(description="Response for feedback based on feedback type")


strParser = StrOutputParser()

class ClassifyFeedbackSchema(BaseModel):
    response:Literal["positive","negative"] = Field(title="Positive or Negative Feedback")


classifyFeedbackSchemaParser = PydanticOutputParser(
    pydantic_object=ClassifyFeedbackSchema
)

class UserNegativeFeedbackDiagnosisSchema(BaseModel):
    issue_type:Literal["UX","Sales","Payment","Support"] = Field(description='The category of issue mentioned in the review')
    tone:Literal['angry',"Frustrated",'Calm'] = Field(description='The category of tone mentioned in the review')
    urgency:Literal['low','medium','high'] = Field(description='The category of urgency mentioned in the review')


negative_feedback_dignosis_schema_parser = PydanticOutputParser(
    pydantic_object=UserNegativeFeedbackDiagnosisSchema
)

def classify_feedback_type(state: FeedbackSchemaState):
    feedback = state.user_feedback
    classifyFeedbackPrompt = PromptTemplate(
        template="""
            Classify this user Prompt {feedback},  in this format \n {format_instructions}
        """,
        input_variables=["feedback"],
        partial_variables={
            "format_instructions": classifyFeedbackSchemaParser.get_format_instructions()
        },

    )

    classify_feedback_chain = classifyFeedbackPrompt | model | classifyFeedbackSchemaParser
    result = classify_feedback_chain.invoke({
        "feedback":feedback,
    })

    return {"feedback_type":result.response}


def reply_with_positive_response(state: FeedbackSchemaState):
    user_feedback = state.user_feedback

    prompt = PromptTemplate(
        template="""
            Based on this feedback {user_feedback} reply the user feedback with positive tone
        """,
        input_variables=["user_feedback"],
    )

    positive_response_chain = prompt | model | strParser

    result = positive_response_chain.invoke({'user_feedback':user_feedback})

    return {"response":result}

def derive_user_feedback_dignosis(state: FeedbackSchemaState):
    feedback = state.user_feedback

    derive_diagnosis_prompt = PromptTemplate(
       template="""
        Derive the negative feedback info from this user Feedback '{user_feedback}', \n {format_instructions}
       """,
        input_variables=["user_feedback"],
        partial_variables={
            "format_instructions": negative_feedback_dignosis_schema_parser.get_format_instructions()
        }
    )

    dignosis_chain = derive_diagnosis_prompt | model | negative_feedback_dignosis_schema_parser

    result = dignosis_chain.invoke({
        "user_feedback":feedback,
    })
    print("RESULT",result)
    return {"diagnosis":result.model_dump()}

def reply_with_negative_response(state: FeedbackSchemaState):

    user_feedback = state.user_feedback
    diagnosis = state.diagnosis

    reply_negative_response_template = PromptTemplate(
        template="""
    You are a customer support representative.

    Customer Feedback:
    {user_feedback}

    Analysis:
    - Urgency: {urgency}
    - Issue Type: {issue_type}
    - Tone: {tone}

    Write a professional response to the customer.

    Requirements:
    - Acknowledge the issue.
    - Apologize if appropriate.
    - Show empathy.
    - Offer assistance.
    - Return ONLY the response text.
    - Do NOT return JSON.
    - Do NOT return markdown.
    """,
        input_variables=[
            "user_feedback",
            "urgency",
            "issue_type",
            "tone",
        ],
    )

    reply_negative_response_chain = reply_negative_response_template | model | strParser

    result = reply_negative_response_chain.invoke({
        "user_feedback":user_feedback,
        "urgency":diagnosis["urgency"],
        "issue_type":diagnosis["issue_type"],
        "tone":diagnosis["tone"],
    })

    return {"response":result}


def determineFeedbackChain(state: FeedbackSchemaState):
    feedback = state.feedback_type

    if feedback == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"





initialGraphState = {
    "user_feedback":"I've using the app for last two months it's worst app",
    "feedback_type":"",
    "diagnosis":{},
    "response":"",
}


# Graph

graph =  StateGraph(
    state_schema=FeedbackSchemaState,
)

# nodes
graph.add_node("classify_user_feedback",classify_feedback_type)

graph.add_node("positive_response",reply_with_positive_response)
graph.add_node("negative_response",reply_with_negative_response)

graph.add_node("run_diagnosis",derive_user_feedback_dignosis)


# edges

graph.add_edge(START,"classify_user_feedback")

# condition edge
graph.add_conditional_edges("classify_user_feedback",determineFeedbackChain)

graph.add_edge("positive_response",END)

graph.add_edge("run_diagnosis","negative_response")
graph.add_edge("negative_response",END)


workflow = graph.compile()

result = workflow.invoke(initialGraphState)


print(result)


