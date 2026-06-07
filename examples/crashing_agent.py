#!/usr/bin/env python3
"""Demo "agent" that emits the noisy output ACK is meant to compress.

Run it through ACK to watch the pruners fire:

    ack run -- python examples/crashing_agent.py

It prints a 200-line log flood (repetition pruner) and then raises a real
exception (traceback pruner). Run it directly to see the uncompressed output.
"""


def level_three() -> None:
    raise KeyError("target key not found")


def level_two() -> None:
    for _ in range(200):
        print("DEBUG: allocating buffer space...")
    level_three()


if __name__ == "__main__":
    level_two()
