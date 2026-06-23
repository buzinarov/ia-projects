"""A small local tool-calling agent, served by Ollama (no external API,
no API key -- keeps this safe to run from a public repo).

Two tools:
  - classify_product: calls our trained vision model on an image + gender
  - search_similar_products: RAG lookup over the product catalog (src/rag.py)

The agent decides which tool(s) to call based on the user's message; this
module just wires the tool-calling loop, it doesn't hardcode when to search
vs. classify.
"""
import json
from pathlib import Path

import ollama
from PIL import Image

from .inference import load_model, predict
from .rag import search_similar_products

MODEL_NAME = "llama3.1:8b"
VISION_MODEL = "proposed"  # the model we'd actually ship

SYSTEM_PROMPT = (
    "You are a helpful shopping assistant for an online fashion catalog. "
    "You have two tools: `classify_product`, which runs a trained vision "
    "model on a product photo to predict its category, and "
    "`search_similar_products`, which searches the catalog by description. "
    "Use them when they would help answer the user, and explain your "
    "reasoning briefly in plain language. Don't make up product details "
    "that didn't come from a tool result.\n\n"
    "Only call `classify_product` if the user's message explicitly includes "
    "an attached image path (look for a line like '[Attached product image "
    "path: ...]'). Never invent, guess, or use a placeholder image path or "
    "URL. If the user asks you to classify a product but no image is "
    "attached, tell them to upload or select one first instead of calling "
    "the tool."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "classify_product",
            "description": (
                "Predict a product's category from a photo and the shopper's "
                "gender segment, using our trained multi-modal model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the product image file"},
                    "gender": {
                        "type": "string",
                        "description": "Gender segment for the product (Men, Women, Boys, Girls, or Unisex)",
                    },
                },
                "required": ["image_path", "gender"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_similar_products",
            "description": "Search the product catalog for items matching a free-text description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language description of what to find"},
                    "n_results": {"type": "integer", "description": "How many matches to return", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]

_vision_model_cache = {}


def _get_vision_model():
    if "model" not in _vision_model_cache:
        model, maps, device = load_model(VISION_MODEL)
        _vision_model_cache["model"] = (model, maps, device)
    return _vision_model_cache["model"]


def _run_classify_product(image_path, gender):
    model, maps, device = _get_vision_model()
    image = Image.open(image_path)
    return predict(model, maps, device, image, gender)


def _run_search_similar_products(query, n_results=5):
    return search_similar_products(query, n_results=int(n_results))


def _execute_tool_call(name, arguments):
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    try:
        if name == "classify_product":
            return _run_classify_product(**arguments)
        if name == "search_similar_products":
            return _run_search_similar_products(**arguments)
        return {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def chat(user_message, history=None, image_path=None, gender=None, max_tool_rounds=4):
    """Runs one turn of the agent loop. `history` is a list of prior
    {"role", "content"} messages (excluding the system prompt)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + (history or [])

    content = user_message
    if image_path:
        content += f"\n\n[Attached product image path: {image_path}, gender segment: {gender}]"
    messages.append({"role": "user", "content": content})

    tool_trace = []
    for _ in range(max_tool_rounds):
        response = ollama.chat(model=MODEL_NAME, messages=messages, tools=TOOLS)
        message = response["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return {"reply": message.get("content", ""), "tool_trace": tool_trace, "messages": messages[1:]}

        for call in tool_calls:
            fn = call["function"]
            result = _execute_tool_call(fn["name"], fn["arguments"])
            tool_trace.append({"tool": fn["name"], "arguments": fn["arguments"], "result": result})
            messages.append({"role": "tool", "content": json.dumps(result)})

    return {"reply": "I wasn't able to finish that within the allowed tool-call steps.",
            "tool_trace": tool_trace, "messages": messages[1:]}


if __name__ == "__main__":
    result = chat("What kind of footwear do you have for men in blue?")
    print(json.dumps(result["tool_trace"], indent=2))
    print()
    print(result["reply"])
