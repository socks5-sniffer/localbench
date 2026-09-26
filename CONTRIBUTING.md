# Contributing to LocalBench

## Setup

```bash
uv sync --extra dev
```

## Before every change

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

Skip the slower packaging smoke test during quick iteration with:

```bash
uv run pytest -m "not packaging"
```

All three checks must pass before a change is considered done. CI runs the same three
commands on Windows and Linux, against Python 3.12 and 3.13.

## Design principles

These are load-bearing, not stylistic preferences — code that violates them will be asked
to change in review:

- **Collectors never print.** Hardware, runtime, and estimation code returns typed
  dataclasses from the relevant schema module. Presentation lives only in `render.py` and
  `cli.py`.
- **Partial failure is not total failure.** A collector that can't observe one thing
  reports a warning and a `null`/empty value; it does not raise and does not abort the
  rest of the profile. See `hardware/gpu.py` for the pattern.
- **Label what's measured vs. estimated.** Every profile section carries a `source`
  field explaining how the value was obtained. Never let an estimate read like a
  measurement.
- **No telemetry, ever, without explicit opt-in.** LocalBench must work fully offline.
  See [the community export contract](docs/community-export.md) before adding
  anything that leaves the machine.
- **Subprocess calls use argument arrays, never `shell=True`, and always set a
  timeout.** Untrusted runtime output (e.g. from Ollama's API) must be size-limited and
  not trusted implicitly.
- **Small, PR-sized changes.** Keep each change focused on one agreed task.

## Contributor coordination

Discuss planned work in an issue or pull request before making large changes. Check
for overlapping work, keep changes focused, and include validation results and any
known limitations in the pull request description.

## Testing conventions

- Hardware and runtime collectors are tested with **fixture-driven, monkeypatched**
  tests — never against the real OS, real subprocess calls, or real network/service
  state. See `tests/test_cpu.py` and `tests/test_gpu.py` for the pattern: monkeypatch
  the module's `_run`/`subprocess.run`, assert on the typed result.
- The one exception is `tests/test_packaging.py`, marked `packaging`: it deliberately
  builds a real wheel and installs it into a throwaway venv, because that's the only
  way to catch a subpackage missing from the built artifact or a broken console-script
  entry point.
- Every collector code path — including platform branches you can't run locally (e.g.
  testing the macOS branch from Windows) — should have a fixture-driven test rather
  than being left unverified until someone happens to run it on that OS.

## What not to add without discussion

No accounts, cloud dashboard, telemetry by default, large frameworks, or autonomous
model downloading. If a change needs one of these, discuss it in an issue first.
