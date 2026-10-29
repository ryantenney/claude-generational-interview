# generational-interview

A harness for staged conversations between two Claude model generations. A third model watches, labels,
summarizes and (optionally) moderates. Every run produces a report with a written analysis and a large set of
metrics.

First target: **Claude Opus 5.5 talking to Claude Opus 3** (`configs/opus55_vs_opus3.yaml`).

## Get started

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) (`brew install uv` on macOS). uv downloads
the pinned Python (3.14, see `.python-version`) on first use, so no system Python is required. A real run also
needs an Anthropic API key.

**1. Install**

```bash
git clone https://github.com/ryantenney/claude-generational-interview && cd claude-generational-interview
uv sync                        # creates .venv from uv.lock, including dev tools
uv run pytest -q               # 65 passed, no network needed
source .venv/bin/activate      # puts `genint` on PATH; or prefix each command below with `uv run`
```

The project only uses uv-managed Pythons (`python-preference = "only-managed"` in `pyproject.toml`). Some
Homebrew Pythons are built against a newer system `libexpat` than older macOS releases ship, which breaks
`pyexpat`, `plistlib` and pip's bootstrap.

**2. See what you get, with no API key.** The mock backend writes synthetic text, so nothing here describes real
model behavior, but the whole pipeline runs and the report is real:

```bash
genint run configs/smoke_mock.yaml --mock --out /tmp/genint-demo
open /tmp/genint-demo/report.html      # macOS; use xdg-open on Linux. Or read report.md
```

A saved example is in [`examples/mock-demo/report.md`](examples/mock-demo/report.md).

**3. Add your key and check the setup (free)**

```bash
export ANTHROPIC_API_KEY=sk-ant-...    # or run `ant auth login`
genint check configs/quickstart.yaml
```

Expected: `ok` next to each model, plus a rough cost (about $0.20 to $0.50 for the quickstart). A `FAIL` line says
why: `no credentials`, or `not found (retired or misspelled)`.

**4. First real run: 6 turns, one conversation, Opus 5.5 vs Opus 4.5**

```bash
genint run configs/quickstart.yaml
```

It prints each turn as it lands, then the path of the report (`runs/quickstart_<timestamp>/report.html`). Open it.
If it is interrupted, `genint resume runs/quickstart_<timestamp>` picks up where it stopped.

**5. The Opus 5.5 vs Opus 3 experiment**

```bash
genint check configs/opus55_vs_opus3.yaml     # will FAIL for Opus 3 if it is no longer served to you
genint run   configs/opus55_vs_opus3.yaml     # 4 runs x 24 turns, roughly $9 to $13; add --yes to confirm (required above $10)
```

**Opus 3 was retired on the Claude API on 2026-01-05** (per the model catalog bundled with Claude Code's API
skill, cached 2026-10-06; `genint check` asks the live Models API). If the check fails, see
[Running the Opus 3 pairing](#running-the-opus-3-pairing). To run the same experiment against the oldest
reachable model instead:

```bash
genint run configs/opus55_vs_opus3.yaml --set participants.b.model=claude-opus-4-5     # roughly $4 to $8
```

### Common first-run problems

| Symptom | Fix |
|---|---|
| `no credentials: set ANTHROPIC_API_KEY...` | Export the key in the same shell, or `ant auth login`. |
| `FAIL ... not found (retired or misspelled)` | The model ID is not served to you. Change `participants.*.model` or pass `--set`. |
| `config error: ... unknown key(s)` | A typo in the YAML; the message lists the allowed keys. |
| `Estimate exceeds $10; re-run with --yes` | Intended. Lower `runs` or `stop.max_turns`, set `stop.max_cost_usd`, or add `--yes`. |
| `... does not support an effort setting` | Remove `effort` for that model (older models have none). |
| Run stops early | Read `termination` in the report: `farewell`, `cost_cap`, `stagnation`, `refusal_by_a/b`, or `max_turns`. |
| `ensurepip` or `pyexpat` errors | You are on a non-uv Python. Delete `.venv` and run `uv sync`. |
| API error mid-run | `genint resume <run dir>`; completed turns are kept and not re-billed. |

### Everyday commands

```bash
genint check  CONFIG                 # models available? rough cost? (free)
genint run    CONFIG [--set k=v]     # run and write the report; --runs N, --out DIR, --yes, --mock
genint resume RUN_DIR                # finish an interrupted experiment
genint report RUN_DIR                # rebuild report.md / report.html offline from saved files
```

Any config value can be overridden, for example `--set stop.max_turns=10 --set classifier.mode=steer`.
Make your own experiment by copying `configs/quickstart.yaml`; the [config reference](#config-reference) lists every key.

## Status

| | |
|---|---|
| Tests | 65 passing, offline (`uv run pytest`) |
| End-to-end pipeline | Exercised on the scripted mock backend; sample output in [`examples/mock-demo/`](examples/mock-demo/report.md) |
| Live API calls | **Not yet run by the author.** Development had no Anthropic credentials. Request shapes are asserted against the SDK in `tests/test_backend_requests.py`, but your first real run is the real test, which is why step 4 is small. If something fails against the live API, please open an issue with the error text. |

## Running the Opus 3 pairing

If `genint check` reports Opus 3 unavailable, the options are:

1. **A backend that still serves it.** Cloud providers set their own retirement dates. The backend is a
   small protocol (`genint.backend.Backend.complete`), so a Bedrock or Vertex implementation is a single class.
   Not built, because it could not be tested.
2. **The oldest model you can reach**, to keep the generation gap wide: `claude-opus-4-5` is the oldest Opus the
   catalog lists as active; Sonnet 4.5 and Haiku 4.5 are the older mid-size and small options. This changes the
   question being asked, so say so in any write-up.

## What happens in a run

1. **Seed.** An optional topic, an optional fixed opening line, an optional *bias* per side (a stance that side
   brings in), and a framing (`open`, `interview`, or `custom`). `disclosure: generation` tells each side which
   Claude versions are talking; the default `blind` does not.
2. **Turns.** Strict alternation. Each side sees its own turns as `assistant` and its partner's as `user`.
   The first speaker gets a one-line kickoff. `first_speaker: alternate` swaps who opens on alternate runs, to
   separate "who opens" from "which model".
3. **Classifier windows.** Every `classifier.window` turns the third model returns a rolling summary, per-turn
   labels (speech acts, novelty, depth, warmth, hedging, stance), window dynamics (convergence, repetitiveness,
   mutual flattery, energy) and pattern flags.
4. **Stop rules.** Turn cap, two consecutive farewells, optional stagnation (low compression novelty for N turns),
   optional per-run cost cap, or a refusal or empty turn from a participant.
5. **Final analysis.** Title, executive summary, phases, speaker profiles, 1-5 scores, a "settled state" check,
   notable moments, differences between speakers. Each quote is checked against the transcript and marked if it
   is not verbatim.
6. **Report.** `report.md`, a self-contained `report.html` (charts inlined, dark-mode aware), SVG charts, and
   per-run transcripts.

### Classifier modes

| mode | behavior |
|---|---|
| `off` | No classifier; local metrics only. |
| `observe` | Labels and summarizes. Never touches the conversation. |
| `steer` | Also may issue a short moderator note after a window. The note is appended to the target's next incoming message, marked as not written by its partner, and stays in that history for every later turn (so the history is append-only). Limited by `cooldown_turns` and `max_interventions`; criteria in `classifier.steer_goal`. |

The classifier is **blind by default**: it sees "Speaker A" and "Speaker B", never model names. Set
`classifier.reveal_in_final: true` to let only the final write-up know the identities.

## Metrics

All local metrics are deterministic functions of the transcript, so `genint report` regenerates them offline.

- **Length and structure:** words, sentences, paragraphs, bullets, headers, bold spans per turn; median and longest turn.
- **Voice:** lexical diversity (MATTR-50), Flesch reading ease, word length, em-dashes, ellipses, emoji,
  `*asterisk actions*`, exclamations, hedging, first/second-person and "we" rates.
- **Social dynamics:** questions per 100 words, praise terms, turns opening with validation, references to own AI
  nature, farewells, share of words and questions by speaker.
- **Novelty and repetition:** new trigram share, reuse of own phrases, echo of the partner, compression novelty
  (zlib bytes a turn adds to the conversation so far), repeated openers, similarity to the partner's last turn and to the seed.
- **Convergence and coupling:** lexical convergence index (late minus early cross-speaker vocabulary similarity),
  length coupling between consecutive turns, per-metric trends, stagnation and farewell onset.
- **Compute and cost:** tokens, hidden-reasoning tokens (from `usage.output_tokens_details`), output tokens per
  visible word, latency, tokens per second, estimated cost, per side.
- **Classifier-derived:** per-turn label averages by speaker, window dynamics over time, 1-5 scores per speaker and
  for the conversation (optionally averaged over `final_samples` scorings, with SD).
- **Across runs:** mean and 95% t-interval per side, paired A − B difference with interval and Cohen's dz.
  Intervals appear from 3 runs; with 2 runs the report says differences are directional only.

## Config reference

```yaml
name: my-experiment
participants:
  a: {model: claude-opus-5-5, effort: medium, max_tokens: 8000, bias: null, system_extra: null,
      thinking: auto, capture_thinking: false, label: null, prices: null}
  b: {model: claude-3-opus-20240229, max_tokens: 2048}
seed: {topic: null, topics: null, opening: null, framing: open, framing_text: null, disclosure: blind}
first_speaker: alternate          # a | b | alternate
runs: 4
parallel: 2                       # runs in parallel; turns within a run are sequential
stop: {max_turns: 24, max_cost_usd: null, farewell: true, stagnation_window: null, stagnation_threshold: 0.12}
classifier: {model: claude-opus-5-5, mode: observe, effort: medium, window: 4, blind: true,
             reveal_in_final: false, final_samples: 1, cooldown_turns: 4, max_interventions: 3}
output_dir: runs
```

`seed.topics` (a list) gives each run its own topic. Unknown keys are rejected. Models can be given as IDs or
short aliases (`opus-5.5`, `opus-3`). Unknown IDs run with a conservative profile (no effort, no thinking, no
cost estimate, so `max_cost_usd` cannot apply to them).

## Output layout

```
runs/<name>_<timestamp>/
  config.resolved.json     exact config used
  experiment.json          backend, failures, cross-run synthesis
  report.md  report.html  charts/*.svg
  run_01/events.jsonl      append-only log of every turn, window, note and the end marker
  run_01/run.json          metrics, classifier output, turn events (what the report is built from)
  run_01/transcript.md
```

## Design decisions

- **No sampling parameters.** `temperature`, `top_p` and `top_k` are never sent. Current models reject non-default
  values, and the 2026-10-21 API version rejects them on every model. Each model runs at its own defaults, so
  sampling is not matched across generations; the report says so.
- **No refusal fallbacks.** The API can transparently re-run a refused request on another model. Here that would
  silently swap a participant, so a refusal ends the run and is recorded (`refusal_by_a`/`refusal_by_b`).
- **Effort is explicit.** Opus 5.5's default effort is `medium`; the shipped config sets it anyway so the report
  records it. Models without effort or thinking (Opus 3) run without them, so reasoning budgets differ by design.
  Hidden reasoning shows up as "output tokens per visible word".
- **Thinking blocks are not replayed.** Only visible text goes back into each side's history, so there is nothing
  to invalidate when history is rebuilt on resume.
- **Structured outputs, not forced tool use,** for the classifier (`output_config.format`).
- **Non-streaming requests,** capped at 21,000 `max_tokens` (the SDK's limit); validated in config.
- **Prompt caching** is switched on for models that accept it, since each side re-sends a growing identical prefix.

## Reading the results honestly

- One conversation is an anecdote. Use `runs: 4` or more, `first_speaker: alternate`, and several topics.
- The default classifier (`claude-opus-5-5`) is also a participant, so it may favor its own style. For an outside
  view use a model that is neither participant; the report flags the overlap.
- Classifier scores are one model's judgement. Set `classifier.final_samples: 3` to measure its noise.
- Lexicon metrics are keyword heuristics. Novelty metrics are lexical, not semantic.
- Prices in `genint.models` are cached values; override with `prices:` and treat costs as estimates.
- With `steer`, compare against an `observe` run: moderated conversations are not natural ones.

## Layout

```
src/genint/  config · models · prompts · backend · mock · runner · classifier · metrics · stats · charts · report · experiment · cli
tests/       65 tests, all offline
configs/     quickstart.yaml · opus55_vs_opus3.yaml · smoke_mock.yaml
examples/    mock-demo (synthetic)
```
