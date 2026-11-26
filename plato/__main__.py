"""CLI.

    plato chat  --workspace ./ws          talk to the agent
    plato watch --workspace ./ws DIR      wake the agent when files land in DIR

Model: PLATO_MODEL="provider:model", e.g. anthropic:claude-opus-5-5,
openai:gpt-5, ollama:llama3.3 (see plato.connect for every provider).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from plato.agent import Agent, Config
from plato.chat import Chat
from plato.connect import connect, providers

DIM, BOLD, CYAN, YELLOW, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[0m"


def _printer(verbose: bool):
    def on_event(kind: str, data: dict) -> None:
        if kind == "wake":
            print(f"{DIM}── wake (budget {data['budget']} steps, {data['inbox']} inbox item(s)) ──{RESET}")
        elif kind == "step" and verbose:
            print(f"{DIM}[step {data['step']}] {data.get('thoughts', '')}{RESET}")
            for ln in (data.get("code") or "").splitlines():
                print(f"{DIM}  │ {ln}{RESET}")
            if data.get("output"):
                for ln in data["output"].rstrip().splitlines()[:15]:
                    print(f"{DIM}  ↳ {ln}{RESET}")
        elif kind == "guard":
            extra = {k: v for k, v in data.items() if k != "name"}
            print(f"{YELLOW}⛨ guard: {data['name']} {extra or ''}{RESET}")
        elif kind == "end":
            print(f"{DIM}── dormant ({data['outcome']}) ──{RESET}")
    return on_event


def _build_agent(args) -> Agent:
    spec = os.environ.get("PLATO_MODEL")
    if not spec:
        sys.exit("set PLATO_MODEL, e.g. anthropic:claude-opus-5-5, openai:gpt-5 or ollama:llama3.3 "
                 f"(providers: {', '.join(providers())})")
    try:
        model = connect(spec)
    except ValueError as e:
        sys.exit(str(e))
    return Agent(
        args.workspace, model,
        config=Config(max_steps=args.max_steps),
        chat=Chat(on_message=lambda text: print(f"{BOLD}{CYAN}agent ›{RESET} {text}")),
        on_event=_printer(args.verbose),
    )


def chat(args) -> None:
    agent = _build_agent(args)
    print(f"{DIM}workspace: {agent.ws.root}  (ctrl-d to quit){RESET}")
    while True:
        try:
            text = input(f"{BOLD}you ›{RESET} ").strip()
        except EOFError:
            print()
            return
        if text:
            agent.send(text)
            agent.wake()


def watch(args) -> None:
    agent = _build_agent(args)
    folder = Path(args.folder).resolve()
    seen = {p.name for p in folder.iterdir()} if folder.exists() else set()
    print(f"{DIM}watching {folder} — drop a file in to wake the agent (ctrl-c to quit){RESET}")
    while True:
        current = {p.name: p for p in folder.iterdir() if p.is_file()}
        for name in sorted(set(current) - seen):
            body = current[name].read_text(encoding="utf-8", errors="replace")[:4000]
            # An honest event on the file's own artifact, not chat in disguise.
            artifact = f"file:{folder / name}"
            agent.follow(artifact, "file")
            agent.log.submit(artifact, actor_type="human", actor_id=agent.human_id, surface="status",
                             op="upload", payload={"label": f"New file {name} in the watched folder",
                                                   "text": body})
            seen.add(name)
        agent.wake()
        time.sleep(args.interval)


def main() -> None:
    p = argparse.ArgumentParser(prog="plato")
    p.add_argument("--workspace", default="./workspace")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("-v", "--verbose", action="store_true", help="show thoughts, code and output per step")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chat")
    w = sub.add_parser("watch")
    w.add_argument("folder")
    w.add_argument("--interval", type=float, default=2.0)
    args = p.parse_args()
    {"chat": chat, "watch": watch}[args.cmd](args)


if __name__ == "__main__":
    main()
