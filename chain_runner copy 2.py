#!/usr/bin/env python3
"""
MLX-LM Long-Chain Runner: Tiered Context Management

Features:
- Handles 100+ steps by NOT passing full history.
- Uses "Tiered Memory" to prioritize critical info.
- auto-calculates token usage to prevent overflows.
"""

import os
import json
from datetime import datetime
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

MODEL_ID = "huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-mlx-4bit"

# Global constraint to ensure we never crash the model or confuse it
# Even if model supports 128k, keeping it under 32k/50k keeps it smarter.
MAX_INPUT_TOKENS = 32000 

INPUTS = {
    "app_name": "NewsFeed",
    "app_idea": "A web app for finding customized news for people to easily customize their news feed.",
    "tech_stack": "PHP, HTML, JavaScript, Tailwind CSS, MySQL for the database."
}

# ---------------------------------------------------------------------
# 2. THE CONTEXT MANAGER CLASS
# ---------------------------------------------------------------------

class ContextManager:
    def __init__(self, tokenizer, global_inputs):
        self.tokenizer = tokenizer
        self.global_inputs = global_inputs
        self.memory = {} # Stores all step outputs {step_key: content}
        self.foundation_keys = [] # Keys that are ALWAYS included (e.g., 'mvp_spec')

    def save_output(self, key, content, is_foundation=False):
        """Saves a step's output. Marks it as foundation if needed."""
        self.memory[key] = content
        if is_foundation:
            self.foundation_keys.append(key)

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def build_smart_prompt(self, step_config):
        """
        Constructs a prompt dynamically, prioritizing:
        1. System Prompt
        2. User Instructions
        3. Foundation Data (Tier 1)
        4. Explicit Dependencies (Tier 2)
        """
        
        # A. Base Components
        sys_msg = step_config['system_prompt']
        
        # formatting the user instruction with global inputs (like app_idea)
        try:
            user_instruction = step_config['instruction_template'].format(**self.global_inputs)
        except KeyError as e:
            user_instruction = f"INSTRUCTION (Error formatting template: {e})"

        # B. Gather Context Layers
        context_blocks = []

        # Layer 1: Foundation (Always include these, e.g., MVP Spec)
        # We skip this if the current step IS the foundation step
        for key in self.foundation_keys:
            if key != step_config['step_key'] and key in self.memory:
                context_blocks.append(f"--- FOUNDATION DOC: {key.upper()} ---\n{self.memory[key]}\n")

        # Layer 2: Explicit Dependencies (Specific requirements for this step)
        for key in step_config.get('dependencies', []):
            if key in self.memory:
                context_blocks.append(f"--- REFERENCE DOC: {key.upper()} ---\n{self.memory[key]}\n")
            else:
                print(f"⚠️ Warning: Dependency '{key}' not found in memory.")

        # C. Assemble and Token Check
        # We build the prompt parts in reverse priority to check size? 
        # Actually, let's just assemble and check.
        
        full_context_text = "\n".join(context_blocks)
        
        final_user_content = f"""
{user_instruction}

AVAILABLE CONTEXT:
{full_context_text}
"""
        
        # Check Token Count
        total_tokens = self.count_tokens(sys_msg) + self.count_tokens(final_user_content)
        
        print(f"   📊 Prompt Token Count: {total_tokens} / {MAX_INPUT_TOKENS}")
        
        if total_tokens > MAX_INPUT_TOKENS:
            print("   ⚠️ WARNING: Context limit exceeded! Truncating context...")
            # Naive truncation for safety - in a real app, you might summarize here
            allowed_chars = len(final_user_content) - ((total_tokens - MAX_INPUT_TOKENS) * 4)
            final_user_content = final_user_content[:allowed_chars] + "\n[...TRUNCATED DUE TO LENGTH...]"

        return sys_msg, final_user_content

# ---------------------------------------------------------------------
# 3. STEP DEFINITIONS (The Chain)
# ---------------------------------------------------------------------

STEPS = [
    {
        "step_key": "mvp_spec",
        "role": "Product Manager",
        "is_foundation": True,  # <--- IMPORTANT: This will now auto-append to ALL future steps
        "dependencies": [],     
        "system_prompt": "You are a Product Manager. Create a lean MVP Spec.",
        "instruction_template": "Create an MVP spec for: {app_idea}"
    },
    {
        "step_key": "db_schema",
        "role": "DBA",
        "is_foundation": True, # Also foundation: DB schema is usually needed everywhere
        "dependencies": [], # It will automatically get 'mvp_spec' because mvp_spec is foundation
        "system_prompt": "You are a DBA. Create a Postgres Schema.",
        "instruction_template": "Create the DB schema based on the MVP."
    },
    {
        "step_key": "landing_page_copy",
        "role": "Copywriter",
        "is_foundation": False, 
        "dependencies": [], # Needs MVP (auto-included), doesn't need DB Schema (irrelevant)
        "system_prompt": "You are a Marketing Expert. Write H1, H2, and CTA copy.",
        "instruction_template": "Write landing page copy for {app_name}."
    },
    {
        "step_key": "api_user_auth",
        "role": "Backend Dev",
        "is_foundation": False,
        "dependencies": [], # Needs DB Schema (auto-included), Needs MVP (auto-included)
        "system_prompt": "Write a Next.js API route for User Auth.",
        "instruction_template": "Write the code for POST /api/auth/signup using {tech_stack}."
    },
    # Example of step 100
    {
        "step_key": "frontend_settings_page",
        "role": "Frontend Dev",
        "is_foundation": False,
        "dependencies": ["api_user_auth"], # Only needs specific API ref + Foundation
        "system_prompt": "Write a React Component.",
        "instruction_template": "Create the Settings Profile page using the Auth API."
    }
]

# ---------------------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------------------

def setup_env():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("outputs", f"run_{ts}")
    os.makedirs(path, exist_ok=True)
    return path

def run_inference(model, tokenizer, sys, user):
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True
    )
    sampler = make_sampler(temp=0.6, top_p=0.9)
    output = generate(model, tokenizer, prompt=prompt, max_tokens=2500, sampler=sampler, verbose=False)
    return output.strip()

def main():
    print(f"🤖 Loading Model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)
    output_dir = setup_env()
    
    # Initialize Context Manager
    ctx_mgr = ContextManager(tokenizer, INPUTS)
    
    for i, step in enumerate(STEPS):
        print(f"\n🔹 STEP {i+1}: {step['step_key']}")
        
        # 1. Build Prompt (Auto-handles Foundation + Dependencies)
        sys_msg, user_msg = ctx_mgr.build_smart_prompt(step)
        
        # 2. Run
        result = run_inference(model, tokenizer, sys_msg, user_msg)
        
        # 3. Save to Memory (Mark as foundation if config says so)
        is_foundation = step.get('is_foundation', False)
        ctx_mgr.save_output(step['step_key'], result, is_foundation)
        
        # 4. Save to File
        filename = f"{i+1:02d}_{step['step_key']}.md"
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(result)
        print(f"  ✅ Saved {filename}")

    # Save manifest
    with open(os.path.join(output_dir, "run_manifest.json"), "w") as f:
        json.dump(ctx_mgr.memory, f, indent=2)

if __name__ == "__main__":
    main()