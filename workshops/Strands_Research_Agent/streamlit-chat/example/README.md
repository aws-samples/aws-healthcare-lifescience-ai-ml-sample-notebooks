# PubMed Research Agent

A life science research assistant that searches and analyzes PubMed Central (PMC) articles using citation analysis.

## Features

- Searches PMC for scientific articles with citation-based ranking
- Reads and summarizes full-text articles
- Prioritizes highly-cited and influential papers
- Returns structured results with PMCID references and URLs

## Tools

- `search_pmc_tool`: Search PMC with citation analysis and ranking
- `read_pmc_tool`: Retrieve and summarize full-text articles

## Deploy
1- Change directory
```bash
cd workshops/Strands_Research_Agent/streamlit-chat 
```
2- Install dev dependencies using uv
```bash
uv sync --dev
```

3- Configure and Deploy the Agent
```bash
cd example

# Configure the agent
uv run agentcore configure -e agent.py

# Deploy to AgentCore Runtime
uv run agentcore launch --auto-update-on-conflict
```

4- Run the application
```bash
cd ..
uv run streamlit run app.py
```

## Usage

The agent responds to scientific questions by:
1. Searching PMC for relevant, highly-cited papers
2. Reading key articles for detailed understanding
3. Extracting and summarizing clinical findings
4. Providing citations with PMC IDs and URLs
