"""The agent loop: talks to the model, runs tools, and waits for the user's approval on risky actions.

How it works:
1. The user's message is added to the history and the model is called.
2. The model either answers (the loop ends) or says "call this tool with these parameters".
3. WE run the tool calls, add their results to the history and call the model again.
4. Some tools (delete, update...) STOP before running: the user gets an approval card and the
   pending call is written to the database. When the user clicks "approve", the loop resumes
   where it left off.

Gemini's SDK can run this loop by itself (automatic function calling). We run it ourselves so
we can both show the tool steps and step in for approvals.

The two engines (Gemini, Ollama) keep the history in different formats. There is one loop;
the engine differences live in two small adapter classes below: how a message is built, how the
model is called, how a tool result is added.

Everything returned to the UI is language-neutral data (tool names, codes, ISO dates).
The wording, in the viewer's language, lives in static/index.html.
"""
import base64
import os
import time
import uuid
from datetime import datetime
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import memory
from tools import FUNCTIONS, HANDLERS, TOOL_SCHEMAS, TZ, describe_action, recall

# "gemini" (cloud, needs an API key) or "ollama" (local, free).
ENGINE = os.getenv("ENGINE", "gemini")

# Short-term memory sent to the model: how many of the last user messages (and the answers in
# between). Older ones stay in the database and on screen, but aren't sent (saves tokens).
KEEP_TURNS = int(os.getenv("KEEP_TURNS", "10"))

MAX_STEPS = 8  # max model calls for one message (guards against endless loops)

# The tool result of a rejected action starts with this. The model sees it and knows nothing
# was done; /history checks it to show the step as "rejected".
REJECTED = "REJECTED:"
# Prefixes used by older (Turkish) histories, so they still display correctly.
_REJECTED_PREFIXES = (REJECTED, "REDDEDİLDİ:")
_ERROR_PREFIXES = ("Error", "Hata")

# UI languages. The UI sends its language; the model uses it as a hint for which language to
# answer in when the user's own message doesn't make it obvious.
LANGUAGES = {"en": "English", "fr": "French", "tr": "Turkish"}


def system_text(lang="en"):
    now = datetime.now(ZoneInfo(TZ))
    return (
        "Your name is Jarvis; you are a personal assistant. If you need to introduce yourself, say Jarvis. "
        "The user is a university student living in Paris. "
        "Always reply in the language of the user's latest message; if it's unclear, use "
        f"{LANGUAGES.get(lang, 'English')} (the language of the user's interface). Be short and clear. "
        f"Now: {now.strftime('%A %Y-%m-%d %H:%M')} ({TZ}). "
        "Resolve expressions like 'tomorrow' or 'Friday' from this. If no duration is given, an event lasts 1 hour. "
        "If critical information is missing (e.g. the time), ask first instead of guessing. "
        "To delete or edit an event, find it with list_events first and take its ID from there. "
        "APPROVAL CARDS: before delete_event, update_event, import_ics (confirm=True) and create_event (force=True) run, "
        "the UI shows the user an approval card. Don't ask for confirmation in text for these; call the tool directly, "
        "and when the card appears, say briefly what will happen. "
        f"If a tool result starts with '{REJECTED}', the user did not approve: nothing was done, don't retry. "
        "If create_event returns a conflict warning, call it again with force=True; the card shows the conflict and the user decides. "
        "If the user states a lasting preference (with words like 'always', 'from now on'), save it with remember; "
        "but don't save it again if it's already in the list below. "
        "If the user sends a photo (ticket, invitation, poster...), read the details on it "
        "(date, time, team/event name, place). If the time is unclear or you want to double-check, "
        "verify it with web_search (and fetch_page if the snippet doesn't have it), then have the user confirm "
        "what you found and call create_event if they agree. "
        "IMPORTANT: times found with web_search may be in UTC/GMT, not Europe/Paris local time. "
        "Prefer the local time from French sources (e.g. 'coup d'envoi', 'heure française'); "
        "if you only found UTC/GMT, convert it to Paris time before passing it to create_event "
        "(UTC+2 in summer time, UTC+1 in winter) and tell the user clearly which time you added as Paris time.\n\n"
        # Long-term memory: saved preferences are always in the instructions, so the model doesn't
        # have to "remember" to call the recall tool.
        f"The user's saved preferences:\n{recall()}"
    )


# ---------------------------------------------------------------- approval gate

def needs_approval(name: str, args: dict) -> bool:
    """Can this tool call run without the user's approval? The check is IN THE CODE: whatever
    the model says, a call for which this returns True never runs until the user clicks "approve"."""
    if name in ("delete_event", "update_event"):
        return True
    if name == "import_ics":
        return bool(args.get("confirm"))  # the preview (confirm=False) is free, the actual import needs approval
    if name == "create_event":
        return bool(args.get("force"))     # a normal add is free, adding despite a conflict needs approval
    return False


def step_info(name, args, result):
    """A tool step for the UI: which tool, its key detail, and how it went (ok / error / rejected).
    The UI turns this into a line like "› searching the web: "..." ✓" in its own language."""
    a, result = args or {}, str(result)
    status = ("rejected" if result.startswith(_REJECTED_PREFIXES)
              else "error" if result.startswith(_ERROR_PREFIXES) else "ok")
    detail = {
        "web_search": a.get("query", ""),
        "fetch_page": urlparse(a.get("url", "")).netloc or a.get("url", ""),
        "list_events": a.get("query", ""),
        "create_event": a.get("title", ""),
    }.get(name, "")
    step = {"tool": name, "detail": detail, "status": status}
    if name == "import_ics":
        step["variant"] = "import" if a.get("confirm") else "preview"
    return step


def _run_tool(name, args):
    fn = HANDLERS.get(name)
    result = fn(**args) if fn else f"Error: unknown tool {name}"
    print(f"[tool] {name}({args}) -> {str(result)[:200]}")
    return str(result)


def _process_calls(calls, results, steps):
    """Runs the tool calls in order. When it reaches one that needs approval it STOPS and
    returns it (the rest are processed after the approval). Returns None when all are done."""
    for name, args in calls[len(results):]:
        if needs_approval(name, args):
            return name, args
        result = _run_tool(name, args)
        results.append([name, result])
        steps.append(step_info(name, args, result))
    return None


# ---------------------------------------------------------------- engine 1: Gemini (cloud)

class GeminiEngine:
    name = "gemini"

    def __init__(self):
        from google import genai
        from google.genai import errors, types
        self.types, self.errors = types, errors
        self.client = genai.Client()
        self.models = [m.strip() for m in os.getenv(
            "GEMINI_MODELS", "gemini-2.5-flash-lite,gemini-3.1-flash-lite,gemini-3.8-flash"
        ).split(",") if m.strip()]

    # Gemini's history is a list of the SDK's own "Content" objects.
    # To store it in SQLite we turn it into plain JSON, and back into objects when reading.
    def load(self, sid):
        return memory.load_history(sid, self.name, lambda data: [self.types.Content.model_validate(c) for c in data])

    def save(self, sid, history):
        def serialize(h):
            data = [c.model_dump(mode="json") for c in h]
            for c in data:  # replace photos (inline_data) with a short note
                c["parts"] = [{"text": memory.IMAGE_PLACEHOLDER} if p.get("inline_data") else p
                              for p in c["parts"]]
            return data
        memory.save_history(sid, self.name, history, serialize)

    def is_user_message(self, c):
        # Tool results also have the "user" role; a real user message has text and no tool result.
        parts = c.parts or []
        return (c.role == "user" and any(p.text for p in parts)
                and not any(p.function_response for p in parts))

    def _user_parts(self, text, image):
        parts = []
        if image:  # image + text in the same message; Gemini reads them together
            parts.append(self.types.Part.from_bytes(data=base64.b64decode(image["data"]), mime_type=image["mime_type"]))
        if text:
            parts.append(self.types.Part.from_text(text=text))
        return parts

    def add_user(self, history, text, image):
        # If the model failed right after a turn's tool results, the history may end with a
        # "user" message. Gemini doesn't like two user messages in a row, so we extend that one.
        if history and history[-1].role == "user":
            history[-1].parts = (history[-1].parts or []) + self._user_parts(text, image)
        else:
            history.append(self.types.Content(role="user", parts=self._user_parts(text, image)))

    def tool_results(self, results, text=None, image=None):
        """Tool results go in a single "user" message. If the user wrote a new message without
        answering a pending card, that message goes in the same package (Gemini expects an answer
        to every tool call)."""
        parts = [self.types.Part.from_function_response(name=n, response={"result": r}) for n, r in results]
        return [self.types.Content(role="user", parts=parts + self._user_parts(text, image))]

    def call_model(self, history, system):
        """Calls the model once. Returns: (message to add to the history, answer text, [(tool, params)])."""
        t = self.types
        config = t.GenerateContentConfig(
            system_instruction=system, tools=FUNCTIONS,
            # Turn off automatic function calling: the SDK must not run the tool itself, just tell us.
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        last_error = None
        for model in self.models:  # if a model is busy/missing, try the next one
            for attempt in range(2):
                try:
                    resp = self.client.models.generate_content(model=model, contents=history, config=config)
                    print(f"[gemini] {model}")
                    content = resp.candidates[0].content if resp.candidates else None
                    if not content or not content.parts:  # empty answer: add empty text so the history stays valid
                        content = t.Content(role="model", parts=[t.Part.from_text(text="")])
                    text = "".join(p.text for p in content.parts if p.text and not p.thought)
                    calls = [(p.function_call.name, dict(p.function_call.args or {}))
                             for p in content.parts if p.function_call]
                    return content, text, calls
                except self.errors.APIError as e:
                    last_error = e
                    if e.code == 404:
                        print(f"[gemini] {model} not found, trying the next one")
                        break
                    if e.code in (429, 500, 502, 503, 504):
                        print(f"[gemini] {model} busy ({e.code}), attempt {attempt + 1}")
                        time.sleep(1 + attempt * 2)
                        continue
                    raise
        raise last_error

    @staticmethod
    def display(raw):
        """Turns the saved history (plain JSON) into what's shown on screen: a list of (kind, data)."""
        out, open_calls = [], []
        for m in raw:
            parts = m.get("parts") or []
            for p in parts:
                if p.get("function_call"):
                    open_calls.append((p["function_call"]["name"], p["function_call"].get("args") or {}))
                elif p.get("function_response") and open_calls:
                    name, args = open_calls.pop(0)
                    out.append(("step", step_info(name, args, (p["function_response"].get("response") or {}).get("result", ""))))
            text = "\n".join(p["text"] for p in parts if p.get("text") and not p.get("thought"))
            if text:
                out.append(("me" if m["role"] == "user" else "bot", text))
        return out


# ---------------------------------------------------------------- engine 2: Ollama (local)

class OllamaEngine:
    name = "ollama"

    def __init__(self):
        import ollama
        self.ollama = ollama
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

    # Ollama's messages are already plain dictionaries; no JSON conversion needed.
    def load(self, sid):
        return memory.load_history(sid, self.name)

    def save(self, sid, history):
        out = []
        for m in history:  # replace photos with a short note
            if m.get("images"):
                m = {**m, "images": None,
                     "content": f"{memory.IMAGE_PLACEHOLDER} {m.get('content') or ''}".strip()}
            out.append(m)
        memory.save_history(sid, self.name, out)

    def is_user_message(self, m):
        return m["role"] == "user"

    def add_user(self, history, text, image):
        history += self.user_message(text, image)

    def user_message(self, text, image):
        msg = {"role": "user", "content": text}
        if image:
            # NOTE: few local models support images AND tool calling together. If you run into
            # problems, use ENGINE=gemini for this feature.
            msg["images"] = [image["data"]]
        return [msg]

    def tool_results(self, results, text=None, image=None):
        msgs = [{"role": "tool", "name": n, "content": r} for n, r in results]
        return msgs + (self.user_message(text, image) if (text or image) else [])

    def call_model(self, history, system):
        messages = [{"role": "system", "content": system}] + history
        resp = self.ollama.chat(model=self.model, messages=messages, tools=TOOL_SCHEMAS)
        # model_dump() also turns nested objects like tool_calls into plain dicts (so they can be stored as JSON).
        msg = resp["message"].model_dump()
        calls = [(c["function"]["name"], c["function"]["arguments"] or {}) for c in msg.get("tool_calls") or []]
        return msg, msg.get("content") or "", calls

    @staticmethod
    def display(raw):
        out, open_calls = [], []
        for m in raw:
            if m["role"] == "tool" and open_calls:
                name, args = open_calls.pop(0)
                out.append(("step", step_info(name, args, m.get("content", ""))))
                continue
            for c in m.get("tool_calls") or []:
                open_calls.append((c["function"]["name"], c["function"].get("arguments") or {}))
            role = {"user": "me", "assistant": "bot"}.get(m["role"])
            if role and m.get("content"):
                out.append((role, m["content"]))
        return out


ENGINES = {"gemini": GeminiEngine, "ollama": OllamaEngine}
_engine = None


def engine():
    """Create the engine on first use (so importing this module doesn't set up an API client)."""
    global _engine
    if _engine is None:
        _engine = ENGINES[ENGINE]()
    return _engine


# ---------------------------------------------------------------- the loop

def run(sid, message=None, image=None, decision=None, card_id=None, lang="en"):
    """Handles one message (or one decision on an approval card).

    Returns: {"reply": text, "steps": [steps], "pending": approval card or None,
              "notice": optional code the UI turns into a sentence ("stale_card", ...)}
    - message: the user's new message
    - decision: "approve" / "reject" (card buttons); card_id says which card was clicked
    - lang: the UI language, a hint for the model's answer language
    """
    eng = engine()
    system = system_text(lang)
    full = eng.load(sid)
    older, history = memory.split_recent(full, eng.is_user_message, KEEP_TURNS)
    pending = memory.load_pending(sid, eng.name)
    steps = []

    def checkpoint():
        # Only save when the history is "consistent": every tool call has its result.
        eng.save(sid, older + history)

    if decision is not None:
        if not pending or pending["card"]["id"] != card_id:
            return {"reply": "", "steps": [], "pending": None, "notice": "stale_card"}
        memory.clear_pending(sid, eng.name)
        calls, results = pending["calls"], pending["results"]
        name, args = calls[len(results)]
        if decision == "approve":
            result = _run_tool(name, args)
        else:
            result = f"{REJECTED} The user did not approve this action; nothing was done."
        results.append([name, result])
        steps.append(step_info(name, args, result))
        stopped = _process_calls(calls, results, steps)  # continue if other tools were requested at the same time
        if stopped:
            return _pause(sid, eng, checkpoint, calls, results, stopped, steps, reply="")
        history += eng.tool_results(results)
        checkpoint()  # the tool ran; keep its result even if the next model call fails

    elif pending:
        # A new message arrived before the card was answered: the pending (and later) calls count
        # as rejected, and the new message goes to the model in the same package as their results.
        memory.clear_pending(sid, eng.name)
        calls, results = pending["calls"], pending["results"]
        for name, args in calls[len(results):]:
            result = f"{REJECTED} The user wrote a new message without approving; nothing was done."
            results.append([name, result])
            steps.append(step_info(name, args, result))
        history += eng.tool_results(results, message, image)
        checkpoint()

    else:
        eng.add_user(history, message, image)

    texts = []  # the model may also write something while calling tools ("let me check..."); collect it all
    for _ in range(MAX_STEPS):
        item, text, calls = eng.call_model(history, system)
        history.append(item)
        if text.strip():
            texts.append(text.strip())
        if not calls:  # no tool calls means the model gave its final answer
            checkpoint()
            reply = "\n\n".join(texts)
            return {"reply": reply, "steps": steps, "pending": None, "notice": None if reply else "empty_reply"}

        results = []
        stopped = _process_calls(calls, results, steps)
        if stopped:
            return _pause(sid, eng, checkpoint, calls, results, stopped, steps, reply="\n\n".join(texts))
        history += eng.tool_results(results)
        checkpoint()

    checkpoint()
    return {"reply": "\n\n".join(texts), "steps": steps, "pending": None, "notice": "too_many_steps"}


def _pause(sid, eng, checkpoint, calls, results, stopped, steps, reply):
    """Stop at a call that needs approval: prepare the card and write the pending work to the
    database. The history now ends with the model's tool call; its result is added after the decision."""
    name, args = stopped
    card = {"id": uuid.uuid4().hex[:12], "tool": name, **describe_action(name, args)}
    memory.save_pending(sid, eng.name, {"calls": calls, "results": results, "card": card})
    checkpoint()
    return {"reply": reply, "steps": steps, "pending": card, "notice": None}


def display_history(sid):
    """For /history: the messages and steps to show, plus the pending approval card if any."""
    eng = engine()
    raw = memory.load_history(sid, eng.name)  # plain JSON
    items = []
    for kind, data in eng.display(raw):
        if kind == "step":
            # Group consecutive steps together (shown as one block above the answer)
            if items and items[-1]["role"] == "steps":
                items[-1]["steps"].append(data)
            else:
                items.append({"role": "steps", "steps": [data]})
        else:
            items.append({"role": kind, "text": data})
    pending = memory.load_pending(sid, eng.name)
    return {"messages": items, "pending": pending["card"] if pending else None}
