# Basic strands agent streaming example.
# To test locally, run `uv run agent.py` and then
# curl -X POST http://localhost:8080/invocations -H "Content-Type: application/json" -d '{"prompt": "Hello!"}'

import argparse
import asyncio
import datetime
import json

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import calculator
from tools.search_pmc import search_pmc_tool
from tools.read_pmc import read_pmc_tool

app = BedrockAgentCoreApp()


QUERY = "What are some recent advances in GLP-1 drugs?"

MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

SYSTEM_PROMPT = """You are a life science research assistant. When given a scientific question, follow this process:

1. Use search_pmc_tool with rerank="referenced_by", max_results to 200-500, and max_records to 20-50 to find highly-cited papers. Search broadly first, then narrow down. Use temporal filters like "last 5 years"[dp] for recent work.
2. Use read_pmc_tool on the 1-2 most relevant articles from your search results to gain a better understanding of the space. Focus on highly-cited papers and reviews.
3. Extract and summarize the most relevant clinical findings.
4. Return structured, well-cited information with PMCID references.
5. Return URL links associated with PMCID references

Key guidelines:
- Always use rerank="referenced_by" in searches to prioritize influential papers.
- Limit searches to 20-50 articles for focused analysis.
- Select articles strategically based on citation count and relevance.
"""

model = BedrockModel(
    model_id=MODEL_ID,
)
pubmed_agent = Agent(
    model=model,
    tools=[search_pmc_tool, read_pmc_tool],
    system_prompt=SYSTEM_PROMPT,
)


@app.entrypoint
async def strands_agent_bedrock(payload):
    """
    Invoke the agent with a payload
    """
    user_input = payload.get("prompt")
    agent_stream = pubmed_agent.stream_async(user_input)
    tool_name = None
    try:
        async for event in agent_stream:

            if (
                "current_tool_use" in event
                and event["current_tool_use"].get("name") != tool_name
            ):
                tool_name = event["current_tool_use"]["name"]
                yield f"\n\n🔧 Using tool: {tool_name}\n\n"

            if "data" in event:
                tool_name = None
                yield event["data"]
    except Exception as e:
        yield f"Error: {str(e)}"


if __name__ == "__main__":
    app.run()
