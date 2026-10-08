"""Offline tests for agent.py: a scripted fake model + fake tools + a temporary database.

No API key, network or Google account needed. Run from anywhere:
    python tests/test_agent.py
"""
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["JARVIS_DB"] = tempfile.mktemp(suffix=".db")  # before importing: tools/memory read it at import

import agent  # noqa: E402
import memory  # noqa: E402

calls_log = []


def fake(name):
    def f(**kw):
        calls_log.append((name, kw))
        return f"{name} ok"
    return f


for n in ["web_search", "list_events", "delete_event", "update_event", "add_task", "create_event"]:
    agent.HANDLERS[n] = fake(n)
agent.describe_action = lambda name, args: {"action": "other", "tool": name, "args": args}


class Script:
    """What the fake model says on each call: (text, [(tool, params)])."""
    def __init__(self):
        self.queue, self.seen = [], []

    def push(self, *steps):
        self.queue.extend(steps)


class FakeOllama(agent.OllamaEngine):
    """Ollama format (plain dicts), scripted model."""
    def __init__(self, script):
        self.script = script

    def call_model(self, history, system):
        self.script.seen.append(list(history))
        text, calls = self.script.queue.pop(0)
        msg = {"role": "assistant", "content": text,
               "tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls] or None}
        return msg, text, calls


class FakeGemini(agent.GeminiEngine):
    """Gemini format (real SDK objects), scripted model."""
    def __init__(self, script):
        from google.genai import types
        self.types, self.script = types, script

    def call_model(self, history, system):
        t = self.types
        self.script.seen.append([c.model_copy(deep=True) for c in history])
        text, calls = self.script.queue.pop(0)
        parts = ([t.Part.from_text(text=text)] if text else []) + \
                [t.Part(function_call=t.FunctionCall(name=n, args=a)) for n, a in calls]
        return t.Content(role="model", parts=parts), text, calls


failed = 0


def check(label, cond):
    global failed
    print(("  ok    " if cond else "  FAIL  ") + label)
    if not cond:
        failed += 1


def calls_and_results(raw):
    """Number of tool calls and tool results in a saved history (Gemini requires them to match)."""
    n_calls = sum(len([p for p in m.get("parts", []) if p.get("function_call")]) + len(m.get("tool_calls") or [])
                  for m in raw)
    n_results = sum(len([p for p in m.get("parts", []) if p.get("function_response")]) + (m["role"] == "tool")
                    for m in raw)
    return n_calls, n_results


for EngineCls in (FakeOllama, FakeGemini):
    print(f"\n=== {EngineCls.__name__} ===")
    script = Script()
    agent._engine = EngineCls(script)
    sid = f"test-{EngineCls.__name__}"

    # 1) Normal flow: one tool call, then an answer
    calls_log.clear()
    script.push(("", [("web_search", {"query": "Ezhel concert Paris 2026"})]), ("28 October.", []))
    r = agent.run(sid, "when is ezhel playing")
    check("one tool step + answer", r["reply"] == "28 October." and len(r["steps"]) == 1)
    check("step data is language-neutral",
          r["steps"][0] == {"tool": "web_search", "detail": "Ezhel concert Paris 2026", "status": "ok"})

    # 2) Delete -> card -> approve
    calls_log.clear()
    script.push(("Deleting it.", [("delete_event", {"event_id": "abc"})]))
    r = agent.run(sid, "delete the exam")
    check("delete paused with a card", r["pending"] and r["pending"]["tool"] == "delete_event")
    check("the tool did NOT run without approval", calls_log == [])
    check("card stored as pending", agent.display_history(sid)["pending"]["id"] == r["pending"]["id"])
    card = r["pending"]["id"]
    r2 = agent.run(sid, decision="approve", card_id="wrong-id")
    check("wrong card id is refused", r2["notice"] == "stale_card" and calls_log == [])
    script.push(("Deleted.", []))
    r = agent.run(sid, decision="approve", card_id=card)
    check("tool ran after approval", calls_log == [("delete_event", {"event_id": "abc"})])
    check("model saw the result and answered", r["reply"] == "Deleted." and r["pending"] is None)
    check("card cleared", agent.display_history(sid)["pending"] is None)
    r2 = agent.run(sid, decision="approve", card_id=card)
    check("a second click on the same card does nothing", r2["notice"] == "stale_card" and len(calls_log) == 1)

    # 3) Reject
    calls_log.clear()
    script.push(("", [("update_event", {"event_id": "abc", "start": "2026-10-09T15:00:00"})]),
                ("OK, I left it as is.", []))
    card = agent.run(sid, "move the class to 3pm")["pending"]["id"]
    r = agent.run(sid, decision="reject", card_id=card)
    check("reject: tool did not run", calls_log == [])
    check("reject: step marked 'rejected'", r["steps"][-1]["status"] == "rejected")

    # 4) Ignore the card and write a new message
    calls_log.clear()
    script.push(("", [("delete_event", {"event_id": "x"})]), ("Sure, looking at your new request.", []))
    agent.run(sid, "delete x")
    r = agent.run(sid, "never mind, what's the weather")
    check("new message: tool did not run", calls_log == [])
    check("new message: pending card removed", agent.display_history(sid)["pending"] is None)
    check("new message: model answered", r["reply"] == "Sure, looking at your new request.")

    # 5) Three tool calls at once, the middle one needs approval
    calls_log.clear()
    script.push(("", [("list_events", {"start": "a", "end": "b"}), ("delete_event", {"event_id": "y"}),
                      ("add_task", {"text": "z"})]),
                ("All done.", []))
    r = agent.run(sid, "do all three")
    check("first tool ran, stopped at the second", [c[0] for c in calls_log] == ["list_events"] and r["pending"])
    r = agent.run(sid, decision="approve", card_id=r["pending"]["id"])
    check("after approval the rest ran too", [c[0] for c in calls_log] == ["list_events", "delete_event", "add_task"])
    check("answer arrived", r["reply"] == "All done.")

    # 6) The approval rules
    check("plain create_event is free", not agent.needs_approval("create_event", {"title": "a"}))
    check("create_event with force=True needs approval", agent.needs_approval("create_event", {"force": True}))
    check("import_ics preview is free", not agent.needs_approval("import_ics", {"confirm": False}))

    # 7) Is the saved history consistent, and are the steps displayed?
    h = agent.display_history(sid)["messages"]
    check("display order me/steps/bot", [m["role"] for m in h][:3] == ["me", "steps", "bot"])
    statuses = [s["status"] for m in h if m["role"] == "steps" for s in m["steps"]]
    check("rejected steps show up in the history", statuses.count("rejected") >= 2)
    n_calls, n_results = calls_and_results(memory.load_history(sid, agent._engine.name))
    check(f"every tool call has a result ({n_calls}={n_results})", n_calls == n_results)

    # 8) If the model fails right after an approved action, the history must stay consistent
    calls_log.clear()
    script.push(("", [("delete_event", {"event_id": "q"})]))
    card = agent.run(sid, "delete q")["pending"]["id"]
    script.queue.clear()  # the next model call raises IndexError (like an API error)
    try:
        agent.run(sid, decision="approve", card_id=card)
        crashed = False
    except IndexError:
        crashed = True
    n_calls, n_results = calls_and_results(memory.load_history(sid, agent._engine.name))
    check("tool result saved despite the model error", crashed and calls_log and n_calls == n_results)
    script.push(("Continuing.", []))
    r = agent.run(sid, "go on")
    check("the chat continues after the error", r["reply"] == "Continuing.")
    if EngineCls is FakeGemini:
        last_user = script.seen[-1][-1]
        check("Gemini: consecutive user messages are merged", last_user.role == "user" and
              any(p.function_response for p in last_user.parts) and any(p.text == "go on" for p in last_user.parts))

print("\n=== system prompt and old histories ===")
st = agent.system_text("fr")
check("prompt is in English with the UI language hint",
      st.startswith("Your name is Jarvis") and "French (the language of the user's interface)" in st)
check("old Turkish 'REDDEDİLDİ' results still show as rejected",
      agent.step_info("delete_event", {}, "REDDEDİLDİ: x")["status"] == "rejected")
check("old Turkish 'Hata' results still show as errors",
      agent.step_info("web_search", {}, "Hata: x")["status"] == "error")

os.remove(os.environ["JARVIS_DB"])
print("\nRESULT:", "all passed" if not failed else f"{failed} failed")
sys.exit(1 if failed else 0)
