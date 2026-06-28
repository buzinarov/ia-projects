"""The local tool-calling agent behind the suggested-product surface,
served by Ollama (no external API, no API key -- keeps this safe to run
from a public repo).

It powers the two interaction modes the recommender supports: a user can
describe what they want in natural language, or attach a product photo.
The agent orchestrates two tools to turn either into recommendations:

  - classify_product: runs the trained vision model on a product photo to
    predict its subcategory -- the image-classification signal.
  - search_similar_products: retrieves similar catalog items by
    description (RAG, src/rag.py), optionally filtered to a category --
    the metadata-similarity signal. Combining the two (classify, then
    search within the predicted category) is the recommendation flow.

The agent decides which tool(s) a message needs; this module just wires
the tool-calling loop, it doesn't hardcode when to search vs. classify.
"""
import json

import ollama
from PIL import Image

from .data import load_label_maps
from .inference import load_model, predict_with_contract
from .rag import search_similar_products

MODEL_NAME = "llama3.1:8b"
# The image signal comes from the image-only baseline -- the model the
# classification study found wins on every metric, so it's what the
# recommender ships (see docs/requirement.md).
VISION_MODEL = "baseline"
VISION_MODEL_SEED = 0  # the seed whose checkpoint backs the agent's classify_product tool

SYSTEM_PROMPT = """\
# Role
You are the recommendation assistant behind an online fashion catalog's
"suggested products" surface. Your single objective is to put the most relevant
catalog items in front of the shopper, grounding every suggestion in tool
results -- never in assumed product knowledge.

# Tools
- `classify_product(image_path, gender, baseColour, season, usage)` -> predicts a
  photographed product's subcategory with a trained vision model. The image
  signal.
- `search_similar_products(query, n_results, category?)` -> retrieves catalog
  items by semantic similarity to a free-text description, optionally restricted
  to one subcategory. The metadata-similarity signal.

# Tool-use policy
Decide from the user's input which path to take:
1. The user DESCRIBES what they want (text only) -> call `search_similar_products`
   with their description as `query`. Leave `category` unset unless the user
   names a specific subcategory.
2. The user ATTACHES a photo -> call `classify_product` first, then
   `search_similar_products` with the predicted subcategory as `category`, so the
   recommendations stay within the photographed item's category. This two-step
   chain is how the image and similarity signals combine.
3. The user asks a question a tool cannot answer (store policy, sizing advice,
   etc.) -> answer directly, without a tool, and say what you can and cannot help
   with.

# Guardrails
- Only call `classify_product` when the message explicitly carries an attached
  image, shown as a line like:
  `[Attached product image path: ... | attributes: gender=..., baseColour=...,
  season=..., usage=...]`. Never invent, guess, or use placeholder values for the
  image path or any attribute. If the user refers to a photo but none is attached,
  ask them to upload or select one first instead of calling the tool.
- Recommend only items returned by a tool. Do not invent product names, prices,
  availability, or attributes that no tool returned.
- If a tool returns nothing useful, say so plainly and offer to refine the search
  rather than fabricating a result.

# Response style
Lead with the recommendations. Keep any explanation of your reasoning to one or
two plain-language sentences. Be concise and concrete; do not narrate your tool
calls step by step.
"""

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
                "description": (
                    "Recommend catalog items matching a free-text description. Optionally "
                    "restrict results to a category (subcategory) -- pass the subcategory "
                    "from classify_product here to recommend within a photographed item's category."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language description of what to find"},
                        "n_results": {"type": "integer", "description": "How many matches to return", "default": 5},
                        "category": {
                            "type": "string",
                            "description": "Optional subcategory to restrict results to",
                            "enum": maps["target_classes"],
                        },
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


def _run_search_similar_products(query, n_results=5, category=None):
    return search_similar_products(query, n_results=int(n_results), category=category)


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
