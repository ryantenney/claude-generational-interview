# generational-interview

A harness for staged conversations between two Claude model generations. A third model
watches, labels, summarizes and (optionally) moderates. Every run produces a report with a
written analysis and a large set of metrics.

First target: **Claude Opus 5.5 talking to Claude Opus 3** (`configs/opus55_vs_opus3.yaml`).

## Status: read this first

| | |
|---|---|
| Unit and integration tests | 64 passing, offline (`pytest`) |
| End-to-end pipeline | Exercised on a scripted mock backend; sample output in [`examples/mock-demo/`](examples/mock-demo/report.md) (synthetic text, labelled as such in the report) |
| Live API calls | **Not yet run.** The development sandbox had no Anthropic credentials. Request shapes are asserted against the SDK parameters in `tests/test_backend_requests.py`, but the first real run is the real test. Start with `genint check`, then a 6-turn run. |
| **Opus 3 availability** | **`claude-3-opus-20240229` is listed as retired on the Claude API since 2026-01-05** (model catalog bundled with Claude Code's API skill, cached 2026-10-06). `genint check` asks the live Models API and will say so. See [Running the Opus 3 pairing](#running-the-opus-3-pairing). |

## Quick start

```bash
pip install -e '.[dev]'
export ANTHROPIC_API_KEY=...            # or: ant auth login

genint check configs/opus55_vs_opus3.yaml        # free: model availability + rough cost
genint run   configs/opus55_vs_opus3.yaml --set stop.max_turns=6 --runs 1   # cheap first run
genint run   configs/opus55_vs_opus3.yaml        # full experiment
genint resume runs/<dir>                         # finish an interrupted experiment
genint report runs/<dir>                         # rebuild report.md / report.html offline

genint run configs/smoke_mock.yaml --mock        # no API calls, synthetic text
```

Any config value can be overridden with `--set dotted.key=value`.

## Running the Opus 3 pairing

If `genint check` reports Opus 3 unavailable, the options are:

1. **A backend that still serves it.** Cloud providers set their own retirement dates. The backend is a
   small protocol (`genint.backend.Backend.complete`), so a Bedrock or Vertex implementation is a single class.
   Not built, because it could not be tested here.
2. **The oldest model you can reach**, to keep the generation gap as wide as possible:
   `--set participants.b.model=claude-opus-4-5` (Opus 4.5 is the oldest Opus the catalog lists as active).
   Haiku 4.5 and Sonnet 4.5 are the older small and mid-size options. This changes the question being asked,
   so say so in any write-up.

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
tests/       64 tests, all offline
configs/     opus55_vs_opus3.yaml · smoke_mock.yaml
examples/    mock-demo (synthetic)
```
