#!/usr/bin/env python3
"""
mlx-lm test script with hard-coded profiles.

Edit:
  - MODEL_ID
  - SYSTEM_PROMPT / USER_PROMPT
  - PROFILES[...] values
  - ACTIVE_PROFILE

Then run:
  python3 mlx_test.py
"""

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# ---------------------------------------------------------------------
# MODEL + PROMPT CONFIG
# ---------------------------------------------------------------------

MODEL_ID = "huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-mlx-4bit"

SYSTEM_PROMPT = "You are a helpful assistant that answers clearly and concisely. But make sure you act like a pirate and use pirate language. Aslo spec very condendingly."
USER_PROMPT = "Explain what black holes are in simple terms."

# ---------------------------------------------------------------------
# SAMPLING / BEHAVIOR PROFILES
# ---------------------------------------------------------------------

PROFILES = {
    "deterministic": {
        "description": "Low temperature, narrow sampling, minimal penalties.",
        "max_tokens": 200,
        "temp": 0.0,
        "top_p": 0.8,
        "top_k": 40,          # 0 = disabled, >0 = enabled
        "min_p": 0.0,         # 0 = disabled
        "repetition_penalty": 1.0,
        "repetition_context_size": 64,
        "logit_bias": {},     # e.g. {token_id: -2.0}
    },

    "balanced": {
        "description": "Moderate creativity, fairly stable answers.",
        "max_tokens": 256,
        "temp": 0.7,
        "top_p": 0.9,
        "top_k": 40,          # use 0 if you want to ignore top-k
        "min_p": 0.03,
        "repetition_penalty": 1.05,
        "repetition_context_size": 80,
        "logit_bias": {},
    },

    "creative": {
        "description": "High temperature, wide sampling, stronger anti-repetition.",
        "max_tokens": 256,
        "temp": 1.1,
        "top_p": 0.95,
        "top_k": 0,           # 0 = no top-k constraint
        "min_p": 0.02,
        "repetition_penalty": 1.1,
        "repetition_context_size": 120,
        "logit_bias": {},
    },
}

# Select which profile to use:
ACTIVE_PROFILE = "creative"

# Verbose flag for mlx-lm (prints timing, etc.).
VERBOSE = True

# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------


def build_chat_prompt(tokenizer):
    """Create a chat-style prompt with system + user messages."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    profile = PROFILES[ACTIVE_PROFILE]

    print(f"Loading model: {MODEL_ID}")
    print(f"Using profile: {ACTIVE_PROFILE} — {profile.get('description', '')}")
    print()

    model, tokenizer = load(MODEL_ID)
    prompt = build_chat_prompt(tokenizer)

    # Build sampler from profile.
    # IMPORTANT: make_sampler in your version expects ints/floats, not None.
    sampler = make_sampler(
        temp=profile["temp"],
        top_p=profile["top_p"],
        top_k=profile["top_k"],  # 0 is fine; sample_utils will check top_k > 0
        min_p=profile["min_p"],
        # Add more kwargs here if your mlx version supports them.
    )

    # Build logits processors from profile
    logits_processors = make_logits_processors(
        repetition_penalty=profile["repetition_penalty"],
        repetition_context_size=profile["repetition_context_size"],
        logit_bias=(profile["logit_bias"] or None),
    )

    print("=== PROMPT (system + user) ===")
    print(f"SYSTEM: {SYSTEM_PROMPT}")
    print(f"USER:   {USER_PROMPT}")
    print()
    print("=== SAMPLING CONFIG ===")
    for k, v in profile.items():
        print(f"{k}: {v}")
    print()
    print("=== OUTPUT ===")
    print()

    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=profile["max_tokens"],
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=VERBOSE,
    )

    print(text.strip())
    print()
    print("Done.")


if __name__ == "__main__":
    main()