#!/usr/bin/env python3
"""
MLX-LM Chain Runner: MVP -> Directory Structure -> Database Schema

Usage:
1. Edit MODEL_ID, APP_IDEA, and TECH_STACK.
2. Select an ACTIVE_PROFILE.
3. Run: python3 chain_runner.py
"""

import os
import time
from datetime import datetime
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# ---------------------------------------------------------------------
# 1. CONFIGURATION & INPUTS
# ---------------------------------------------------------------------

# The Model to use (Must be an MLX format model)
MODEL_ID = "huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-mlx-4bit"

# YOUR INPUTS HERE
APP_IDEA = "A web app for finding customized news for people to easily customize their news feed."
TECH_STACK = "PHP, HTML, JavaScript, Tailwind CSS, MySQL."

# ---------------------------------------------------------------------
# 2. PROMPT CHAIN DEFINITIONS
# ---------------------------------------------------------------------

PROMPTS = {
    "step_1": {
        "role": "Product Manager",
        "system": """You are an expert Product Manager and MVP Specialist. Your goal is to strip a web app idea down to its absolute bare essentials.
When provided an app idea, generate a "MVP Specification" containing ONLY:
1. **The Core Loop**: One sentence on the specific action providing value.
2. **Target User**: Narrowly defined audience.
3. **Must-Have Features**: Limit 3-5 critical features. Mark others as "Post-MVP".
4. **User Flow**: Numbered list from Landing Page to Success State.
5. **Data Entities**: Simple list of nouns (e.g., Users, Posts).
CRITICAL: Do not include "nice-to-haves" like settings or profiles unless critical to the core value.""",
    },
    "step_2": {
        "role": "System Architect",
        "system": """You are a Senior Software Architect. You will be given an "MVP Specification" and a "Tech Stack".
Task: Generate a text-based directory structure.
1. **Standard Conventions**: Adhere strictly to the provided Tech Stack best practices.
2. **Separation**: Separate frontend, backend, and config.
3. **Format**: Use a text-based tree format.
4. **Annotation**: Add short comments next to key directories.
""",
    },
    "step_3": {
        "role": "Database Administrator",
        "system": """You are a Lead Database Administrator. You will be given an "MVP Specification".
Task: Design the database schema.
1. **Schema Definition**: Write clearly in SQL (PostgreSQL) or Prisma Schema.
2. **Relationships**: Define FKs (1-1, 1-N, N-N).
3. **Critical Fields**: Include id, created_at, updated_at for every table.
4. **Optimizations**: Suggest indices for the User Flow.
Do not write backend logic, only the schema.
""",
    }
}

# ---------------------------------------------------------------------
# 3. SAMPLING PROFILES
# ---------------------------------------------------------------------

PROFILES = {
    "precise": {
        "description": "Strict, low temp. Good for JSON/Code structures.",
        "max_tokens": 2048,
        "temp": 0.1,
        "top_p": 0.85,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.1,
        "repetition_context_size": 64,
        "logit_bias": {},
    },
    "balanced": {
        "description": "Good mix of creativity and structure.",
        "max_tokens": 2048,
        "temp": 0.6,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
        "repetition_context_size": 100,
        "logit_bias": {},
    }
}

ACTIVE_PROFILE = "balanced"  # Recommended for architectural reasoning
VERBOSE = True # Print token generation speed info

# ---------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ---------------------------------------------------------------------

def setup_output_dir():
    """Creates a unique timestamped directory for this run."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("outputs", f"run_{ts}")
    os.makedirs(path, exist_ok=True)
    print(f"📂 Output directory created: {path}")
    return path

def save_to_file(directory, filename, content):
    """Saves text content to a markdown file."""
    filepath = os.path.join(directory, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"💾 Saved: {filepath}")

def build_prompt(tokenizer, system_content, user_content):
    """Applies the chat template."""
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def run_inference(model, tokenizer, prompt_text, profile_name):
    """Runs the generation logic."""
    p = PROFILES[profile_name]
    
    sampler = make_sampler(
        temp=p["temp"],
        top_p=p["top_p"],
        top_k=p["top_k"],
        min_p=p["min_p"]
    )
    
    logits_processors = make_logits_processors(
        repetition_penalty=p["repetition_penalty"],
        repetition_context_size=p["repetition_context_size"],
        logit_bias=(p["logit_bias"] or None),
    )

    output = generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=p["max_tokens"],
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=VERBOSE,
    )
    return output.strip()

# ---------------------------------------------------------------------
# 5. MAIN EXECUTION LOOP
# ---------------------------------------------------------------------

def main():
    print(f"🤖 Loading Model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)
    
    output_dir = setup_output_dir()
    
    # --- STEP 1: MVP OVERVIEW ---
    print("\n🔹 STARTING STEP 1: MVP OVERVIEW")
    sys_prompt = PROMPTS["step_1"]["system"]
    user_input = f"APP IDEA: {APP_IDEA}"
    
    full_prompt = build_prompt(tokenizer, sys_prompt, user_input)
    mvp_output = run_inference(model, tokenizer, full_prompt, ACTIVE_PROFILE)
    
    save_to_file(output_dir, "1_mvp_overview.md", f"# MVP Overview\n\n{mvp_output}")

    # --- STEP 2: DIRECTORY STRUCTURE ---
    print("\n🔹 STARTING STEP 2: DIRECTORY STRUCTURE")
    sys_prompt = PROMPTS["step_2"]["system"]
    # Context Injection: passing the output of step 1
    user_input = f"CONTEXT (MVP SPEC):\n{mvp_output}\n\nTECH STACK PREFERENCE:\n{TECH_STACK}"
    
    full_prompt = build_prompt(tokenizer, sys_prompt, user_input)
    arch_output = run_inference(model, tokenizer, full_prompt, ACTIVE_PROFILE)
    
    save_to_file(output_dir, "2_directory_structure.md", f"# Directory Structure\n\n{arch_output}")

    # --- STEP 3: DATABASE SCHEMA ---
    print("\n🔹 STARTING STEP 3: DATABASE SCHEMA")
    sys_prompt = PROMPTS["step_3"]["system"]
    # Context Injection: passing the output of step 1 (Stack not strictly needed for logic, but helpful)
    user_input = f"CONTEXT (MVP SPEC):\n{mvp_output}\n\nBased on the user flow in the MVP spec, design the schema."
    
    full_prompt = build_prompt(tokenizer, sys_prompt, user_input)
    db_output = run_inference(model, tokenizer, full_prompt, "precise") # Force 'precise' for SQL
    
    save_to_file(output_dir, "3_database_schema.md", f"# Database Schema\n\n{db_output}")

    print("\n✅ Chain Completed Successfully.")
    print(f"   Outputs located in: {output_dir}")

if __name__ == "__main__":
    main()
