"""The routing agent: the "chatbot" that turns a free-text message from an
agent or customer into a call to one of the four skills.

Served by a local Ollama model (no external API, no key -- safe to run from
a public repo), exactly like the classifier project's agent. The model's job
is narrow: read the message, pick the right skill, and extract that skill's
arguments from the text. The skills (src/skills.py) do the actual NLP.

A keyword fallback router is included so the dispatch path works -- and the
app and tests run -- even when no Ollama server is reachable. The fallback is
deliberately dumb; the Ollama path is the real one.
"""
import json
import os

from .contract import SKILLS
from .skills import SKILL_FUNCS

# Override with a smaller model if 8B won't fit in memory, e.g.
#   set CRI_OLLAMA_MODEL=llama3.2:3b   (PowerShell: $env:CRI_OLLAMA_MODEL="llama3.2:3b")
MODEL_NAME = os.environ.get("CRI_OLLAMA_MODEL", "llama3.1:8b")

SYSTEM_PROMPT = """\
# Role
You are the routing assistant for "Car-ing is sharing", an auto dealership. You
help support agents and customers by dispatching each message to exactly one
specialized tool. You do NOT answer from your own knowledge -- you route.

# Tools
- `triage(review)` -> classify the sentiment of a customer review.
- `translate(text)` -> translate English text to Spanish for a customer.
- `answer(question, context)` -> answer a question using only the text of one
  review (the `context`). Extract the review the user pasted as `context`.
- `digest(text)` -> summarize a long review into a couple of sentences.

# Policy
- Pick the single best tool and extract its arguments from the user's message.
- For `answer`, the `context` must be the review/passage the user provided; do
  not invent one. If the user asks a question but pastes no review to ground it
  in, ask them to paste the review instead of calling the tool.
- If no tool fits (e.g. small talk, a policy question), reply directly in one
  short sentence and say which of triage/translate/answer/digest you can do.

# Style
Be concise. Do not narrate your tool calls.
"""


# --- Tool schema for the Ollama tool-calling loop ------------------------

def _build_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "triage",
                "description": "Classify the sentiment (positive/negative) of a customer car review.",
                "parameters": {
                    "type": "object",
                    "properties": {"review": {"type": "string", "description": "The review text to classify"}},
                    "required": ["review"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "translate",
                "description": "Translate English text to Spanish for a Spanish-speaking customer.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string", "description": "The English text to translate"}},
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "answer",
                "description": "Answer a question using only the text of a single review provided as context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to answer"},
                        "context": {"type": "string", "description": "The review text to find the answer in"},
                    },
                    "required": ["question", "context"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "digest",
                "description": "Summarize a long review into a couple of sentences.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string", "description": "The review text to summarize"}},
                    "required": ["text"],
                },
            },
        },
    ]


def dispatch(skill, arguments):
    """Run one skill by name with the arguments the router extracted.

    Pure dispatch -- no LLM -- so it is unit-testable on its own. Returns the
    skill's contract record, or an {"error": ...} dict for an unknown skill or
    bad arguments.
    """
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if skill not in SKILLS:
        return {"error": f"Unknown skill: {skill}"}
    try:
        return SKILL_FUNCS[skill](**arguments)
    except TypeError as exc:
        return {"error": f"Bad arguments for {skill}: {exc}"}
    except Exception as exc:  # surface model/runtime errors instead of crashing the app
        return {"error": f"{type(exc).__name__}: {exc}"}


# --- Keyword fallback router (used when Ollama is unreachable) ------------

def keyword_route(message):
    """Cheap rule-based routing: returns (skill, arguments).

    A stand-in for the LLM router so the app and tests still demonstrate
    dispatch offline. Order matters: a question wins over a bare summarize
    request. The extraction is naive on purpose -- the Ollama path is what
    does this well.
    """
    low = message.lower()
    if "translate" in low or "spanish" in low or "español" in low:
        return "translate", {"text": _strip_command(message)}
    # Sentiment intent wins over the bare "?" rule: "Is this review positive?"
    # is a triage request, not a question to answer from a context.
    if any(w in low for w in ("sentiment", "positive", "negative", "happy", "unhappy")):
        return "triage", {"review": _strip_command(message)}
    if "summar" in low or "digest" in low or "tl;dr" in low or "tldr" in low:
        return "digest", {"text": _strip_command(message)}
    if "?" in message or low.startswith(("what", "why", "how", "did", "does", "was")):
        return "answer", {"question": message.strip(), "context": _strip_command(message)}
    return "triage", {"review": _strip_command(message)}


def _strip_command(message):
    """Drop a leading 'translate:'/'summarize:' style prefix, keep the payload."""
    for sep in (":", " - ", "—"):
        if sep in message:
            head, _, tail = message.partition(sep)
            if len(head) < 30 and tail.strip():
                return tail.strip()
    return message.strip()


# --- The full agent loop (Ollama) ----------------------------------------

def _ollama_available():
    try:
        import ollama

        ollama.list()
        return True
    except Exception:
        return False


def chat(user_message, history=None, use_ollama=True):
    """One turn of the assistant. Returns
    {"skill", "arguments", "record", "reply", "via"}.

    `via` is "ollama" or "keyword" so the app/UI can show which router ran.
    Falls back to the keyword router automatically if Ollama is unreachable.
    """
    if use_ollama and _ollama_available():
        try:
            return _chat_ollama(user_message, history)
        except Exception as exc:
            # Ollama is reachable but the call failed at runtime -- most often
            # the model won't fit in memory (OOM on load). Don't error out;
            # degrade to the keyword router so the user still gets an answer.
            return _keyword_reply(user_message, via=f"keyword (ollama failed: {type(exc).__name__})")
    return _keyword_reply(user_message, via="keyword")


def _keyword_reply(user_message, via):
    skill, arguments = keyword_route(user_message)
    record = dispatch(skill, arguments)
    return {
        "skill": skill,
        "arguments": arguments,
        "record": record,
        "reply": _summarize_record(skill, record),
        "via": via,
    }


def _chat_ollama(user_message, history):
    import ollama

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + (history or [])
    messages.append({"role": "user", "content": user_message})
    response = ollama.chat(model=MODEL_NAME, messages=messages, tools=_build_tools())
    message = response["message"]

    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return {"skill": None, "arguments": None, "record": None,
                "reply": message.get("content", ""), "via": "ollama"}

    fn = tool_calls[0]["function"]
    skill, arguments = fn["name"], fn["arguments"]
    record = dispatch(skill, arguments)
    return {
        "skill": skill,
        "arguments": arguments,
        "record": record,
        "reply": _summarize_record(skill, record),
        "via": "ollama",
    }


def _summarize_record(skill, record):
    """Render a skill's contract record as a one-line human reply."""
    if "error" in record:
        return f"Sorry, I couldn't run {skill}: {record['error']}"
    p = record["payload"]
    if skill == "triage":
        return f"Sentiment: {p['sentiment']} (confidence {p['confidence']:.2f})."
    if skill == "translate":
        return f"Spanish: {p['translated_text']}"
    if skill == "answer":
        return f"Answer: {p['answer']}"
    if skill == "digest":
        return f"Summary: {p['summary']}"
    return "Done."


if __name__ == "__main__":
    out = chat("Is this review positive? 'Absolutely love this car, best purchase ever.'",
               use_ollama=False)
    print(json.dumps({k: v for k, v in out.items() if k != "record"}, indent=2))
