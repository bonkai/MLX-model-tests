# MLX Long-Chain Runner

Runs very long, multi-step LLM generation chains (100+ steps) locally on Apple Silicon
via MLX — without blowing the context window.

## The problem it solves

Naively passing full history into each step overflows context and degrades quality. This
runner uses **tiered memory**: it prioritizes critical state, summarizes or drops the
rest, and auto-calculates token usage to stay under a safe input budget (default 32k)
even on models that nominally support more — keeping the model fast and coherent across
long chains.

## Stack

- Python, [`mlx-lm`](https://github.com/ml-explore/mlx-lm) (Apple Silicon)
- Local quantized models (e.g. Qwen3-Next 80B 4-bit)

## Run

```bash
pip install mlx-lm
python chain_runner.py
```

> `mlx_test.py` is a minimal smoke test for the MLX generation setup.
