from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents.middleware import wrap_model_call, ModelRequest
from typing import Callable
import os
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPEN_AI_API_KEY")

thinking_model = ChatOpenAI(
    model="gpt-5.1-2025-11-13",
    api_key=OPENAI_API_KEY,
    reasoning_effort="medium",
)

standard_model = ChatOpenAI(
    model="gpt-4.1-2025-04-14",
    api_key=OPENAI_API_KEY,
    temperature=0.0,
)

@wrap_model_call
def choose_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], "ModelResponse"],
):
    """
    Baca request.runtime.context["mode"] yang dikirim dari Streamlit.

    - "thinking"  -> pakai gpt-5
    - "standard"  -> pakai gpt-4.1 (default)
    """
    mode = request.runtime.context.get("mode", "standard")

    if mode == "thinking":
        request.model = thinking_model
    else:
        request.model = standard_model

    return handler(request)