"""A small local tool-calling agent, served by Ollama (no external API,
no API key -- keeps this safe to run from a public repo).

Two tools:
  - classify_product: calls our trained vision model on an image + the
    product's structured attributes (gender, baseColour, season, usage)
  - search_similar_products: RAG lookup over the product catalog (src/rag.py)

The agent decides which tool(s) to call based on the user's message;
this module just wires the tool-calling loop, it doesn't hardcode when
to search vs. classify.
"""
import json

import ollama
from PIL import Image

from .data import load_label_maps
from .inference import load_model, predict_with_contract
from .rag import search_similar_products

MODEL_NAME = "llama3.1:8b"
VISION_MODEL = "proposed"  # the model we'd actually ship
VISION_MODEL_SEED = 0  # the seed whose checkpoint backs the agent's classify_product tool

SYSTEM_PROMPT = (
    "You are a helpful shopping assistant for an online fashion catalog. "
    "You have two tools: `classify_product`, which runs a trained vision "
    "model on a product photo plus its structured attributes (gender, "
    "baseColour, season, usage) to predict its subcategory, and "
    "`search_similar_products`, which searches the catalog by description. "
    "Use them when they would help answer the user, and explain your "
    "reasoning briefly in plain language. Don't make up product details "
    "that didn't come from a tool result.\n\n"
    "Only call `classify_product` if the user's message explicitly includes "
    "an attached image path and its attributes (look for a line like "
    "'[Attached product image path: ... | attributes: gender=..., "
    "baseColour=..., season=..., usage=...]'). Never invent, guess, or use "
    "placeholder values for the image path or any attribute. If the user "
    "asks you to classify a product but no image is attached, tell them to "
    "upload or select one first instead of calling the tool."
)

_label_maps_cache = {}
_vision_model_cache = {}


def _get_label_maps():
    if "maps" not in _label_maps_cache:
        _label_maps_cache["maps"] = load_label_maps()
    return _label_maps_cache["maps"]


def _get_vision_model():
    if "model" not in _vision_model_cache:
        _vision_model_cache["model"] = load_model(VISION_MODEL, seed=VISION_MODEL_SEED)
    return _vision_model_cache["model"]


def _build_tools(maps):
    attr_props = {
        col: {
            "type": "string",
            "description": f"The product's {col} attribute",
            "enum": maps["attribute_classes"][col],
        }
        for col in maps["attribute_columns"]
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "classify_product",
                "description": (
                    "Predict a product's subcategory from a photo and its structured "
                    "attributes, using our trained multi-modal model."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "Path to the product image file"},
                        **attr_props,
                    },
                    "required": ["image_path", *maps["attribute_columns"]],
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


def _run_classify_product(image_path, **attrs):
    model, maps, device = _get_vision_model()
    image = Image.open(image_path)
    missing = [c for c in maps["attribute_columns"] if c not in attrs]
    if missing:
        return {"error": f"Missing required attributes: {missing}"}
    record = predict_with_contract(
        model, maps, device, image,
        {col: attrs[col] for col in maps["attribute_columns"]},
        model_name=VISION_MODEL,
    )
    return record


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


def chat(user_message, history=None, image_path=None, attrs=None, max_tool_rounds=4):
    """Runs one turn of the agent loop. `history` is a list of prior
    {"role", "content"} messages (excluding the system prompt). `attrs`
    is a dict covering all of label_maps.json's attribute_columns,
    required only if `image_path` is set."""
    maps = _get_label_maps()
    tools = _build_tools(maps)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + (history or [])

    content = user_message
    if image_path:
        attrs = attrs or {}
        attr_str = ", ".join(f"{col}={attrs.get(col, 'unknown')}" for col in maps["attribute_columns"])
        content += f"\n\n[Attached product image path: {image_path} | attributes: {attr_str}]"
    messages.append({"role": "user", "content": content})

    tool_trace = []
    for _ in range(max_tool_rounds):
        response = ollama.chat(model=MODEL_NAME, messages=messages, tools=tools)
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
