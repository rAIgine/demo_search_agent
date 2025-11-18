# fmcg_agents.py
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from dotenv import load_dotenv

from shared_state import choose_model, thinking_model, standard_model
import os
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPEN_AI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# 1. Shared model
# model = init_chat_model("gpt-4.1", model_provider="openai", temperature=0, api_key=OPENAI_API_KEY)
# model = ChatOpenAI(model="gpt-5.1-2025-11-13",
#                     # model="gpt-4.1-2025-04-14",
#                     api_key=OPENAI_API_KEY,
#                     # temperature=0.0,
#                     reasoning_effort="low",
#                 )


# 2. Shared Tavily tool (bisa juga bikin 2: general + finance/news)
tavily_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
    api_key=TAVILY_API_KEY
)

# 3. Prompts untuk masing-masing metric
GDP_PROMPT = """
You are an economic analyst focused on GDP and GDP growth.
Task:
- When asked about GDP or GDP growth for a region/country, use Tavily Search.
- Prefer official macro sources (World Bank, IMF, national statistics, OECD).
- Return: 
  - latest GDP level (total and/or per capita),
  - GDP growth rate (YoY) if available,
  - year of data,
  - 2-3 bullet points on what this implies for FMCG demand (volume & value).
Always include 2-4 high quality sources with URLs.
"""

INFLATION_PROMPT = """
You are a macro analyst focused on inflation.
Task:
- Find latest headline inflation (CPI) and if possible food & fuel inflation for the region.
- Return: level (%), trend (rising/falling), and time period.
- Explain what this means for FMCG: 
  - high inflation reduces volume but may increase nominal value.
Prioritize official stats (central bank, national statistics, World Bank, IMF).
"""

FUEL_PROMPT = """
You are a logistics and cost analyst focused on fuel and oil prices.
Task:
- Find current retail fuel prices relevant to consumer goods distribution 
  (e.g. diesel, petrol RON95/92 depending on country).
- If needed, compare with 6-12 months ago to detect trend.
- Explain implications for distribution cost, routing, and allocation priority.
"""

SENTIMENT_PROMPT = """
You are a consumer sentiment analyst.
Task:
- Find latest consumer confidence / sentiment index for the region or country.
- Summarize whether sentiment is optimistic/neutral/pessimistic.
- Explain implications for FMCG: discretionary vs staple categories.
Use reputable sources (central bank, Nielsen, OECD, World Bank, reliable news).
"""

POP_PROMPT = """
You are a demographics analyst.
Task:
- Retrieve latest population level and growth rate for the region,
  and if available, population density.
- Highlight urban vs rural split if possible.
- Explain how population size/growth affects baseline FMCG demand.
"""

URBAN_PROMPT = """
You are an urbanisation analyst.
Task:
- Retrieve urbanization rate (percent of population living in urban areas) 
  and its trend (rising / stable / falling).
- Highlight major urban centers if mentioned.
- Explain how urbanization level affects per capita FMCG consumption and modern trade vs general trade.
"""

# 4. Build agents
gdp_agent = create_agent(
    standard_model,
    tools=[tavily_search],
    system_prompt=GDP_PROMPT,
    name='gdp_agent',
    middleware=[choose_model]
)

inflation_agent = create_agent(
    standard_model,
    tools=[tavily_search],
    system_prompt=INFLATION_PROMPT,
    name='inflation_agent',
    middleware=[choose_model]
)

fuel_agent = create_agent(
    standard_model,
    tools=[tavily_search],
    system_prompt=FUEL_PROMPT,
    name='fuel_agent',
    middleware=[choose_model]
)

sentiment_agent = create_agent(
    standard_model,
    tools=[tavily_search],
    system_prompt=SENTIMENT_PROMPT,
    name='sentiment_agent',
    middleware=[choose_model]
)

population_agent = create_agent(
    standard_model,
    tools=[tavily_search],
    system_prompt=POP_PROMPT,
    name='population_agent',
    middleware=[choose_model]
)

urbanization_agent = create_agent(
    standard_model,
    tools=[tavily_search],
    system_prompt=URBAN_PROMPT,
    name='urbanization_agent',
    middleware=[choose_model]
)
