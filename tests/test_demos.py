"""The demos in demos/ work end to end, driven through their real interfaces
(HTTP API, terminal) with the offline stand-in model."""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from demos._shared.models import OfflineModel
from demos._shared.web import start_in_background
from demos.browser_chat.__main__ import build as build_chat
from demos.comment_section.__main__ import build as build_comments

ROOT = Path(__file__).resolve().parent.parent


def wait_until(pred, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        value = pred()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("timed out")


class Client:
    def __init__(self, app, agent):
        self.server = app.serve(port=0)
        start_in_background(self.server)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.stop = threading.Event()
        threading.Thread(target=agent.serve, kwargs={"stop": self.stop, "poll": 0.1}, daemon=True).start()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as r:
            return json.loads(r.read())

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    def events(self):
        return self.get("/api/state?after=0")["events"]

    def close(self):
        self.stop.set()
        self.server.shutdown()


def test_hello_cli_answers_and_takes_follow_ups_mid_thought(tmp_path):
    env = {**os.environ, "PLATO_OFFLINE_DELAY": "1"}
    env.pop("PLATO_MODEL", None)
    proc = subprocess.Popen([sys.executable, "-m", "demos.hello_cli", "--workspace", str(tmp_path / "ws")],
                            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    proc.stdin.write("hello there\n")
    proc.stdin.flush()
    time.sleep(0.5)
    proc.stdin.write("and a follow-up\n")
    proc.stdin.flush()
    time.sleep(5)
    out = proc.communicate(timeout=10)[0]  # closes stdin itself (closing it first breaks on 3.12)
    assert "I read: “hello there”" in out
    assert "Also got: “and a follow-up”" in out                      # joined the same wake
    assert out.count("awake") == 1


def test_browser_chat_replies_and_reactions(tmp_path):
    agent, app = build_chat(str(tmp_path / "ws"), OfflineModel(delay=0))
    c = Client(app, agent)
    try:
        c.post("/api/send", {"text": "plan a trip to Lisbon"})
        answer = wait_until(lambda: [e for e in c.events() if e["actor_type"] == "agent" and "I read" in e["text"]])[0]
        mine = c.post("/api/send", {"text": "book the first option", "reply_to": answer["id"]})
        wait_until(lambda: [e for e in c.events() if e["parent_id"] == mine["id"] and e["actor_type"] == "agent"])
        n = len([e for e in c.events() if e["actor_type"] == "agent"])
        c.post("/api/react", {"id": answer["id"], "emoji": "👍"})
        time.sleep(0.5)
        assert len([e for e in c.events() if e["actor_type"] == "agent"]) == n   # 👍 is logged only
        c.post("/api/react", {"id": answer["id"], "emoji": "❓"})
        wait_until(lambda: len([e for e in c.events() if e["actor_type"] == "agent"]) > n)
        reasons = {g["reason"] for g in c.get("/api/state")["gates"]}
        assert {"chat-message", "no-rule", "reaction-❓"} <= reasons
    finally:
        c.close()


def test_comment_section_wakes_only_when_tagged(tmp_path):
    agent, comments, app = build_comments(str(tmp_path / "ws"), OfflineModel(delay=0))
    c = Client(app, agent)
    try:
        c.post("/api/comment", {"author": "jonas", "text": "Made this, loved it."})
        c.post("/api/comment", {"author": "lena", "text": "@jonas whole wheat?"})
        time.sleep(0.6)
        assert not [e for e in c.events() if e["actor_type"] == "agent"]  # neither woke it
        tag = c.post("/api/comment", {"author": "maria", "text": "@plato can I skip the Dutch oven?"})
        reply = wait_until(lambda: [e for e in c.events() if e["actor_type"] == "agent" and e["surface"] == "comment"])[0]
        assert reply["parent_id"] == tag["id"]                           # answered in maria's thread
        eyes = [e for e in c.events() if e["surface"] == "reaction" and e["actor_type"] == "agent"]
        assert [(e["op"], e["parent_id"]) for e in eyes] == [("👀", tag["id"])]
        follow = c.post("/api/comment", {"author": "aiko", "text": "@plato and if my kitchen is cold?",
                                         "reply_to": tag["id"]})
        wait_until(lambda: len([e for e in c.events() if e["actor_type"] == "agent" and e["surface"] == "comment"]) == 2)
        replies = [e for e in c.events() if e["actor_type"] == "agent" and e["surface"] == "comment"]
        assert replies[1]["parent_id"] == tag["id"]                      # same thread
        assert any(e["op"] == "👀" and e["parent_id"] == follow["id"] for e in c.events())
        assert set(c.get("/api/state")["woke"]) == {tag["id"], follow["id"]}
    finally:
        c.close()
