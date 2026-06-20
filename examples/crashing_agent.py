#!/usr/bin/env python3
"""Demo "agent" that emits the noisy output ACK is meant to compress.

Run it through ACK to watch both pruners fire:

    ack run -- python examples/crashing_agent.py

It simulates an agent doing a build-and-test run: a few status lines pass through
untouched, a 200-line log flood collapses to a frequency table, and a deep
test-failure traceback (two app frames buried under framework/stdlib noise)
collapses to the exception plus the user frames. The short pauses between phases
keep it readable (and make a nice screen recording); run it directly, without
ACK, to see the full uncompressed output for comparison.
"""
import sys
import time


def say(message: str, pause: float = 0.9) -> None:
    """Print a status line the way an agent would, then pause briefly."""
    print(message)
    sys.stdout.flush()
    time.sleep(pause)


def emit_log_flood(n_lines: int = 200) -> None:
    for _ in range(n_lines):
        print("DEBUG: allocating buffer space...")
    sys.stdout.flush()


def emit_buried_traceback(n_noise_frames: int = 40) -> None:
    """Relay a realistic traceback: 2 app frames under a wall of framework noise."""
    frames = [
        "Traceback (most recent call last):",
        '  File "/app/src/views/checkout.py", line 201, in post',
        "    order = cart.checkout(user_id=user.pk, payment=payload)",
        '  File "/app/src/models/cart.py", line 88, in checkout',
        "    charge_result = gateway.charge(amount, card_token)",
    ]
    for i in range(n_noise_frames):
        frames.append(
            '  File "/home/user/.venv/lib/python3.11/site-packages/django/core/'
            f'handlers/base.py", line {200 + i}, in _get_response'
        )
        frames.append("    response = wrapped_callback(request, *args, **kwargs)")
    frames.append("KeyError: 'card_token'")
    print("\n".join(frames))
    sys.stdout.flush()


if __name__ == "__main__":
    say("[agent] Building project and running the test suite...", 1.2)
    say("[agent] Compiling dependencies (verbose output follows)...", 1.0)
    emit_log_flood()
    time.sleep(1.4)
    say("[agent] Compile done. Running pytest...", 1.2)
    emit_buried_traceback()
    time.sleep(1.0)
    say("[agent] Test run failed (see compressed traceback above).", 0.6)
