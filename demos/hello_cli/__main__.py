"""Hello world: an always-on agent in your terminal.

    python -m demos.hello_cli

The agent runs in the background and sleeps until you write. Keep typing while
it's thinking: follow-ups land in the same wake, at its next step.
"""

from __future__ import annotations

import argparse
import threading

from demos._shared.models import describe, make_model, spend_limits
from plato import Agent, Chat

try:
    import readline  # line editing + keeping your half-typed line when the agent speaks
except ImportError:  # e.g. Windows
    readline = None

DIM, BOLD, CYAN, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[0m"
PROMPT = f"{BOLD}you ›{RESET} "


def say(line: str) -> None:
    """Print above the prompt without eating what you're typing."""
    typed = readline.get_line_buffer() if readline else ""
    print(f"\r\033[K{line}\n{PROMPT}{typed}", end="", flush=True)


def on_event(kind: str, data: dict) -> None:
    if kind == "wake":
        say(f"{DIM}· awake ({data['inbox']} new){RESET}")
    elif kind == "step":
        say(f"{DIM}· step {data['step']}: {data.get('thoughts') or '…'}{RESET}")
    elif kind == "end":
        say(f"{DIM}· dormant{RESET}")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m demos.hello_cli")
    p.add_argument("--workspace", default="demos/hello_cli/workspace")
    args = p.parse_args()

    model = make_model()
    agent = Agent(
        args.workspace, model, **spend_limits(model),
        chat=Chat(on_message=lambda text: say(f"{BOLD}{CYAN}agent ›{RESET} {text}")),
        on_event=on_event,
    )
    stop = threading.Event()
    threading.Thread(target=agent.serve, kwargs={"stop": stop}, daemon=True).start()

    print(f"{DIM}plato hello world · {describe(model)} · workspace {agent.ws.root}\n"
          f"Type a message. Keep typing while it thinks. Ctrl-D to quit.{RESET}")
    try:
        while True:
            text = input(PROMPT).strip()
            if text:
                agent.send(text)  # just an event on the log; the agent picks it up
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        stop.set()


if __name__ == "__main__":
    main()
