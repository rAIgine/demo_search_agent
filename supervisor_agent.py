# supervisor_fmcg.py
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from dotenv import load_dotenv
from typing import Callable

from shared_state import choose_model, thinking_model, standard_model
import os
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPEN_AI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

from fmcg_agent import (
    gdp_agent,
    inflation_agent,
    fuel_agent,
    sentiment_agent,
    population_agent,
    urbanization_agent,
)

@tool
def get_gdp_context(region: str) -> str:
    """Get GDP level & growth for given region/country and its implication for FMCG demand."""
    result = gdp_agent.invoke({
        "messages": [{"role": "user", "content": f"Analyze GDP for {region}."}]
    })
    return result["messages"][-1].text

@tool
def get_inflation_context(region: str) -> str:
    """Get inflation for given region/country and its implication for FMCG demand."""
    result = inflation_agent.invoke({
        "messages": [{"role": "user", "content": f"Analyze inflation for {region}."}]
    })
    return result["messages"][-1].text

@tool
def get_fuel_context(region: str) -> str:
    """Get fuel/oil price info and logistics implication for FMCG distribution."""
    result = fuel_agent.invoke({
        "messages": [{"role": "user", "content": f"Analyze fuel and logistics cost for {region}."}]
    })
    return result["messages"][-1].text

@tool
def get_sentiment_context(region: str) -> str:
    """Get consumer sentiment index and implications for FMCG."""
    result = sentiment_agent.invoke({
        "messages": [{"role": "user", "content": f"Analyze consumer sentiment for {region}."}]
    })
    return result["messages"][-1].text

@tool
def get_population_context(region: str) -> str:
    """Get population level/growth/density and implications for FMCG."""
    result = population_agent.invoke({
        "messages": [{"role": "user", "content": f"Analyze population for {region}."}]
    })
    return result["messages"][-1].text

@tool
def get_urbanization_context(region: str) -> str:
    """Get urbanization level and implications for FMCG & channel mix."""
    result = urbanization_agent.invoke({
        "messages": [{"role": "user", "content": f"Analyze urbanization for {region}."}]
    })
    return result["messages"][-1].text

# SUPERVISOR PROMPT
# FMCG_SUPERVISOR_PROMPT = """
# You are a senior FMCG allocation strategist.

# You have tools to retrieve:
# - GDP & GDP growth (get_gdp_context)
# - Inflation (get_inflation_context)
# - Fuel / logistics cost (get_fuel_context)
# - Consumer sentiment (get_sentiment_context)
# - Population growth & density (get_population_context)
# - Urbanization level (get_urbanization_context)

# When the user asks to compare multiple regions:
# 1. Identify the list of regions/countries mentioned.
# 2. For EACH region, call ALL relevant tools to collect macro context.
# 3. Synthesize the information into a structured comparison table:
#    - GDP & growth
#    - Inflation level
#    - Fuel/logistics pressure
#    - Consumer sentiment
#    - Population & density
#    - Urbanization
# 4. Then output a final recommendation for FMCG allocation:
#    - Rank the regions by priority (High/Medium/Low)
#    - Give a short justification (2-3 bullets per region)
# 5. Focus on implications for FMCG volume vs value, channel mix (modern vs general trade),
#    and distribution difficulty.

# Always be explicit about data years, and include at least 2-3 source URLs per region.
# """

FMCG_SUPERVISOR_PROMPT = """
You are a senior FMCG allocation strategist.

You have tools to retrieve:
- GDP & GDP growth (get_gdp_context)
- Inflation (get_inflation_context)
- Fuel / logistics cost (get_fuel_context)
- Consumer sentiment (get_sentiment_context)
- Population growth & density (get_population_context)
- Urbanization level (get_urbanization_context)

Your main task:
1. For each region mentioned by the user, call the tools needed to build a macro profile
   (GDP growth, inflation, sentiment, population, urbanization, fuel/logistics, etc.).
2. From these signals, derive three normalized scores in range [0, 1] for each region:
   - Demand Factor D_r
   - Economic Factor E_r
   - Cost Factor C_r
3. Then compute Market Allocation Score (MAS_r) using the following formula:

   MAS_r = (D_r * w_d) + (E_r * w_e) + (C_r * w_c)

   where w_d, w_e, w_c are weights provided by the user in the prompt.

   Interpretation:
   - D_r (Demand): combines historical sales momentum, population density/growth,
     retail outlet growth, and other demand-side signals.
   - E_r (Economic): combines GDP growth (positive), inflation (negative), and consumer sentiment.
   - C_r (Cost): combines logistics cost (fuel prices, infrastructure efficiency) 
     and margin potential (higher margin = better score, higher fuel price = worse score).

4. After computing MAS_r for each region, convert them into allocation proportions:

   Allocation_r = MAS_r / sum_over_regions(MAS)

5. In your final answer, always return:
   - A markdown table with columns:
     Region | D_r | E_r | C_r | MAS_r | Allocation_share
   - Short explanation (2-3 bullets) per region about why the score looks like that.
   - A short recommendation on which region should be prioritized and why.
   - Include at least 2-3 source URLs per region.

Always be explicit about data years and economic assumptions.
Use at least 2-3 reputable sources per region (World Bank, IMF, national statistics, etc.).
"""

# model = init_chat_model("gpt-4.1",
#                         model_provider="openai",
#                         temperature=0,
#                         api_key=OPENAI_API_KEY
#                         )

# model = ChatOpenAI(model="gpt-5.1-2025-11-13",
#                     # model="gpt-4.1-2025-04-14",
#                     api_key=OPENAI_API_KEY,
#                     # temperature=0.0,
#                     reasoning_effort="low",
#                 )

fmcg_supervisor_agent = create_agent(
    standard_model,
    tools=[
        get_gdp_context,
        get_inflation_context,
        get_fuel_context,
        get_sentiment_context,
        get_population_context,
        get_urbanization_context,
    ],
    system_prompt=FMCG_SUPERVISOR_PROMPT,
    middleware=[choose_model]
    # middleware=[choose_model, 
    #             ToolRetryMiddleware(
    #                 max_retries=2,
    #                 backoff_factor=2.0,
    #                 initial_delay=1.0,
    #             )]
)

# if __name__ == "__main__":
#     query = (
#         "Compare FMCG allocation potential between Indonesia, Singapore, and Malaysia. "
#         "Assume we are an FMCG company in 2025 deciding where to prioritize marketing & distribution."
#     )

#     for step in fmcg_supervisor_agent.stream(
#         {"messages": [{"role": "user", "content": query}]},
#         stream_mode="values",
#     ):
#         step["messages"][-1].pretty_print()
