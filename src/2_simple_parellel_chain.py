from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langgraph.constants import START,END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

model = ChatOllama(
    model="qwen2.5-coder:7b"
)

strParser = StrOutputParser()

class CricketStatsState(BaseModel):
    runs:int = Field(...,description="Total number of runs")
    balls:int = Field(...,description="Balls faced by Batsman")
    fours:int = Field(...,description="Four hit by Batsman")
    sixes:int = Field(...,description="Sixes faced by Batsman")


    strike_rate:float = Field(...,description="Strike rate")
    balls_per_boundary:float = Field(...,description="Balls Interval")
    boundary_percentage:float = Field(...,description="Boundary percentage")
    summary:str = Field(...,description="Summary of the Batsman")


def calculateStrikerate(state:CricketStatsState):
    runs = state.runs
    balls = state.balls
    strike_rate = (runs / balls) * 100 if balls > 0 else 0

    return {"strike_rate":strike_rate}

def calculateBallsPerBoundary(state:CricketStatsState):
    balls = state.balls
    fours = state.fours
    sixes = state.sixes
    total_boundaries = fours + sixes
    balls_per_boundary = balls / total_boundaries if total_boundaries > 0 else 0

    return {"balls_per_boundary":balls_per_boundary}

def calculateBoundaryPercentage(state:CricketStatsState):
    runs = state.runs
    fours = state.fours
    sixes = state.sixes
    boundary_runs = (fours * 4) + (sixes * 6)
    boundary_percentage = (boundary_runs / runs) * 100 if runs > 0 else 0

    return {"boundary_percentage":boundary_percentage}

def calculateSummary(state:CricketStatsState):
    prompt = PromptTemplate(
        template='''
        You are a cricket analytics expert.

Analyze the batsman using the provided batting metrics and generate a concise professional summary describing the batsman's playing style, aggression level, scoring pattern, boundary dependency, and likely batting role.

Input:
- Runs: {runs}
- Balls Faced: {balls}
- Fours: {fours}
- Sixes: {sixes}
- Strike Rate: {strike_rate}
- Balls Per Boundary: {balls_per_boundary}
- Boundary Percentage: {boundary_percentage}

Derive insights such as:
- Aggressive batsman
- Anchor batsman
- Power hitter
- Finisher
- Boundary-dependent player
- Strike rotator
- Balanced batsman
- Defensive player
- Six-hitting specialist
- High-risk attacking batsman
- Stable innings builder

Analysis Guidelines:
- Use strike rate to judge intent and aggression.
- Use balls per boundary to understand attacking frequency.
- Use boundary percentage to identify dependency on boundaries.
- Consider fours vs sixes distribution to identify batting style.
- Infer batting role naturally (opener, anchor, finisher, aggressor, stabilizer, etc.).
- Mention strengths and possible limitations if visible.
- Avoid repeating raw statistics directly.
- Keep the response analytical, natural, and concise.
- Output should sound like commentary from a professional cricket analyst.

Output:
Return only the final batting summary paragraph.''',
        input_variables=["runs","balls","fours","sixes","strike_rate","balls_per_boundary","boundary_percentage"],
    )

    summaryChain = prompt | model | strParser

    summary = summaryChain.invoke({
        "runs":state.runs,
        "balls":state.balls,
        "fours":state.fours,
        "sixes":state.sixes,
        "strike_rate":state.strike_rate,
        "balls_per_boundary":state.balls_per_boundary,
        "boundary_percentage":state.boundary_percentage,
    })

    return {"summary":summary}

graph = StateGraph(
    state_schema=CricketStatsState
)

# Graphs
graph.add_node("calculate_str",calculateStrikerate)
graph.add_node("calculate_boundary_percentage",calculateBoundaryPercentage)
graph.add_node("calculate_balls_per_boundary",calculateBallsPerBoundary)
graph.add_node("calculate_summary",calculateSummary)


# Edges
graph.add_edge(START,"calculate_str")
graph.add_edge(START,"calculate_boundary_percentage")
graph.add_edge(START,"calculate_balls_per_boundary")

graph.add_edge("calculate_str","calculate_summary")
graph.add_edge("calculate_boundary_percentage","calculate_summary")
graph.add_edge("calculate_balls_per_boundary","calculate_summary")

graph.add_edge("calculate_summary",END)


cricketGraph = graph.compile()

result = cricketGraph.invoke({"runs": 78,
    "balls": 52,
    "fours": 7,
    "sixes": 3,

    # placeholder values (will be calculated by graph)
    "strike_rate": 0.0,
    "balls_per_boundary": 0.0,
    "boundary_percentage": 0.0,
    "summary": ""})


print(f"Strike Rate: {result['strike_rate']} \nBalls Per Boundary: {result['balls_per_boundary']} \nBoundary Percentage: {result['boundary_percentage']} \nSummary: {result['summary']}")















