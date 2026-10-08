import os

from dotenv import load_dotenv
load_dotenv()  # loads the keys in .env (must happen BEFORE any API client is created)

from flask import Flask, jsonify, request, send_from_directory

import agent
from tools import upcoming_events

# This file is only the web side: the addresses (endpoints) the browser talks to.
# The agent loop (model + tools + approval cards) lives in agent.py.
ENGINE = agent.ENGINE

app = Flask(__name__)


def _lang(data):
    """The UI language sent by the browser ("en", "fr" or "tr"); English if missing or unknown."""
    lang = (data or {}).get("lang", "en")
    return lang if lang in agent.LANGUAGES else "en"


def _error(e, where):
    print(f"[{where}] error: {e}")
    return jsonify(reply="", steps=[], pending=None, notice="server_error", detail=str(e)), 500


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/history")
def history():
    """What the UI shows for a chat: messages, tool steps, and the approval card waiting for an
    answer, if any. The model's internal thoughts are not shown."""
    return jsonify(agent.display_history(request.args.get("session_id", "")))


@app.get("/events/upcoming")
def events_upcoming():
    """Upcoming events (JSON), used by the "Next" card on the welcome screen (?limit=3) and the
    agenda panel (?days=30&limit=50). If the calendar can't be reached (no internet, expired
    token...) returns 503; the UI hides the card or shows "try again"."""
    # Upper bounds, so a request like ?days=100000 doesn't turn into a huge call to Google.
    days = min(max(request.args.get("days", 60, type=int), 1), 90)
    limit = min(max(request.args.get("limit", 3, type=int), 1), 100)
    try:
        return jsonify(events=upcoming_events(days, limit))
    except Exception as e:
        print(f"[events] couldn't read the calendar: {e}")
        return jsonify(events=[], error=str(e)), 503


@app.post("/chat")
def chat():
    """A new message. Answer: {"reply": text, "steps": [tool steps], "pending": approval card or null,
    "notice": optional code}"""
    data = request.get_json()
    image = data.get("image")  # {"mime_type": "...", "data": "<base64>"} or nothing
    try:
        return jsonify(agent.run(data["session_id"], data.get("message", ""), image, lang=_lang(data)))
    except Exception as e:
        return _error(e, "chat")


@app.post("/confirm")
def confirm():
    """The approval card's buttons. Body: {"session_id", "card_id", "decision": "approve"|"reject", "lang"}"""
    data = request.get_json()
    if data.get("decision") not in ("approve", "reject"):
        return jsonify(reply="", steps=[], pending=None, notice="invalid_decision"), 400
    try:
        return jsonify(agent.run(data["session_id"], decision=data["decision"],
                                 card_id=data.get("card_id"), lang=_lang(data)))
    except Exception as e:
        return _error(e, "confirm")


if __name__ == "__main__":
    PORT = int(os.getenv("PORT", "5001"))
    print(f"Engine: {ENGINE}")
    # With debug=True Flask runs two processes: a "watcher" that monitors the code and the actual
    # server (restarted on every code change). Only the watcher opens the browser; otherwise you'd
    # get two tabs on start and one more on every code change. The server has WERKZEUG_RUN_MAIN="true".
    # Timer: wait 1.5 s for the server to come up. Disable with OPEN_BROWSER=0.
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true" and os.getenv("OPEN_BROWSER", "1") == "1":
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    app.run(port=PORT, debug=True)
