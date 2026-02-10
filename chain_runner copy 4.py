#!/usr/bin/env python3
"""
MLX-LM Autonomous Developer (Ultra-Robust Edition)
--------------------------------------------------
Model: Qwen 2.5/Next 80B (Recommended)
Strategy: Recursive Generation with AI Peer Review.
Context: Accumulates FULL codebase in context (up to limits).
"""

import os
import json
import logging
import re
import time
from datetime import datetime
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

# Your Model Path
MODEL_ID = "huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-mlx-4bit"

INPUTS = {
    "project_name": "NewsFeed",
    "app_idea": "A web app for finding customized news for people to easily customize their news feed.",
    "tech_stack": "PHP, HTML, JavaScript, Tailwind CSS, MySQL."
}
# Project Settings
PROJECT_NAME = INPUTS["project_name"]
APP_IDEA = INPUTS["app_idea"]
TECH_STACK = INPUTS["tech_stack"]

# Robustness Settings
MAX_RETRIES = 5           # How many times to fix code before giving up
REVIEWER_TEMP = 0.1       # Low temp for strict reviewing
CODER_TEMP = 0.2          # Low temp for precise coding
MAX_CONTEXT_TOKENS = 65000 # Safety limit (even if model does 262k, python slows down)

# =============================================================================
# 2. LOGGING & SETUP
# =============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join("outputs", f"project_{timestamp}")
CODE_DIR = os.path.join(OUTPUT_DIR, "src")
os.makedirs(CODE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "system.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutoDev")

# =============================================================================
# 3. THE VIRTUAL FILE SYSTEM (MEMORY)
# =============================================================================

class ProjectState:
    def __init__(self):
        self.file_manifest = []     # List of dicts {path, type, description}
        self.created_files = {}     # Dict {path: code_content}
        self.full_context_str = ""  # The massive string fed to the LLM

    def update(self, path, code):
        """Adds a new file to the state and updates the context string."""
        self.created_files[path] = code
        # Append to the master context so future files see this code
        entry = f"\n\n--- START FILE: {path} ---\n{code}\n--- END FILE: {path} ---\n"
        self.full_context_str += entry
        logger.info(f"💾 State Updated: Added {path} to global context.")

state = ProjectState()

# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def extract_json(text):
    """Robust JSON extraction."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match: return match.group(1)
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match: return match.group(1)
    return None

def extract_code(text):
    """Extracts code block, handling generic markdown wrappers."""
    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if match: return match.group(1).strip()
    # Fallback: if the model didn't use blocks but just dumped code (rare in 80B)
    return text.strip()

def run_llm(model, tokenizer, system_prompt, user_prompt, temp=0.2, max_tokens=2000):
    """Generic wrapper for MLX generation."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    sampler = make_sampler(temp=temp, top_p=0.9)
    
    # We don't stream to console to keep logs clean, we just wait.
    output = generate(model, tokenizer, prompt=prompt_input, max_tokens=max_tokens, sampler=sampler, verbose=False)
    return output.strip()

# =============================================================================
# 5. AGENT STAGES
# =============================================================================

def stage_1_architect(model, tokenizer):
    """Generates the file manifest."""
    logger.info("🏗️  STAGE 1: ARCHITECT - Planning the build...")
    
    sys = "You are a Senior Software Architect. Break down the app into a file structure."
    prompt = f"""
PROJECT: {PROJECT_NAME}
IDEA: {APP_IDEA}
STACK: {TECH_STACK}

Task: Output a JSON object containing a list of ALL files needed.
Rules:
1. Include ALL config files (package.json, tsconfig.json, requirements.txt).
2. Separate components, pages, hooks, and types.
3. STRICT JSON FORMAT.

Format:
{{
  "files": [
    {{ "path": "package.json", "type": "config", "desc": "Dependencies" }},
    {{ "path": "src/types/index.ts", "type": "types", "desc": "Global interfaces" }}
  ]
}}
"""
    response = run_llm(model, tokenizer, sys, prompt, temp=0.5, max_tokens=2000)
    json_str = extract_json(response)
    
    if not json_str:
        logger.critical("Architect failed to produce JSON. Exiting.")
        exit(1)
        
    try:
        data = json.loads(json_str)
        state.file_manifest = data['files']
        logger.info(f"✅ Plan Approved: {len(state.file_manifest)} files to build.")
        
        # Save plan
        with open(os.path.join(OUTPUT_DIR, "project_plan.json"), "w") as f:
            json.dump(data, f, indent=2)
            
    except json.JSONDecodeError:
        logger.critical("Architect produced invalid JSON.")
        exit(1)

def sort_dependency_order():
    """Sorts files so we build types/configs first, then utils, then UI."""
    # Priority map (Lower # = Build First)
    priority = {
        "config": 1,
        "types": 2,
        "utils": 3,
        "hooks": 4,
        "components": 5,
        "pages": 6,
        "backend": 7
    }
    
    def get_score(f):
        # Explicit type check
        p = priority.get(f.get("type", ""), 10)
        # Path fallback check
        path = f['path'].lower()
        if "tsconfig" in path or "package.json" in path: return 0
        if "types" in path or "interface" in path: return 2
        if "lib" in path or "util" in path: return 3
        if "component" in path: return 5
        return p

    state.file_manifest.sort(key=get_score)
    logger.info("🔄 Re-ordered build queue based on dependency logic.")

def stage_2_builder_loop(model, tokenizer):
    """The Main Loop: Write -> Review -> Fix -> Save."""
    logger.info("🔨 STAGE 2: CONSTRUCTION - Starting the build loop.")
    
    total_files = len(state.file_manifest)
    
    for idx, file_info in enumerate(state.file_manifest):
        path = file_info['path']
        desc = file_info['desc']
        
        logger.info(f"[{idx+1}/{total_files}] Processing: {path}...")
        
        # 1. GENERATE
        code = generate_file_code(model, tokenizer, path, desc)
        
        # 2. REVIEW & FIX
        valid_code = review_and_fix(model, tokenizer, path, code)
        
        # 3. SAVE
        if valid_code:
            full_path = os.path.join(CODE_DIR, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(valid_code)
            
            # 4. UPDATE CONTEXT
            state.update(path, valid_code)
        else:
            logger.error(f"❌ Failed to build {path} after multiple retries. Skipping.")

def generate_file_code(model, tokenizer, path, desc):
    """Generates the initial code draft."""
    sys = "You are an Expert Developer. Write clean, working code."
    
    # We construct the prompt carefully
    prompt = f"""
PROJECT: {PROJECT_NAME}
CURRENT TASK: Write code for file '{path}'
DESCRIPTION: {desc}
STACK: {TECH_STACK}

EXISTING CODEBASE CONTEXT:
{state.full_context_str[-50000:]} 
(Note: Context truncated to last 50k chars for efficiency, but assume previous files exist)

INSTRUCTIONS:
1. Output ONLY the code for this file.
2. Use imports relative to the file structure provided in the plan.
3. Ensure types match the 'types' definitions in the context.
4. If this is package.json, include strict versions.
"""
    return run_llm(model, tokenizer, sys, prompt, temp=CODER_TEMP, max_tokens=3000)

def review_and_fix(model, tokenizer, path, code):
    """
    The 'Self-Healing' Agent.
    It looks at the code and the file structure and checks for errors.
    """
    attempts = 0
    current_code = extract_code(code)
    
    while attempts < MAX_RETRIES:
        attempts += 1
        
        # THE REVIEW PROMPT
        sys = "You are a Senior Code Reviewer. You are extremely strict."
        prompt = f"""
We are reviewing the file: '{path}'

CURRENT CODE:
```
{current_code}
```

FILE MANIFEST (What exists in the project):
{[f['path'] for f in state.file_manifest]}

YOUR TASK:
Check for these specific errors:
1. Syntax errors (unclosed brackets, bad indentation).
2. Import paths (e.g., importing from '../ui/button' when the file is actually at 'src/components/ui/button').
3. Placeholder comments (e.g., "// logic goes here").
4. Missing exports.

DECISION:
If the code is perfect, output ONLY the word "PASS".
If there are errors, output "FAIL" followed by a numbered list of issues to fix.
"""
        review_output = run_llm(model, tokenizer, sys, prompt, temp=REVIEWER_TEMP, max_tokens=500)
        
        if "PASS" in review_output.upper() and len(review_output) < 20:
            logger.info(f"   ✅ Review Passed (Attempt {attempts})")
            return current_code
        
        # If we are here, it failed.
        logger.warning(f"   ⚠️ Review Failed (Attempt {attempts}). Issues: {review_output[:100]}...")
        
        # RE-GENERATION PROMPT (The "Fixer")
        fix_sys = "You are a Developer fixing bugs reported by a reviewer."
        fix_prompt = f"""
FILE: {path}

ORIGINAL CODE:
```
{current_code}
```

REVIEWER FEEDBACK:
{review_output}

TASK: Rewrite the FULL file code to fix these errors. Output only the code.
"""
        new_output = run_llm(model, tokenizer, fix_sys, fix_prompt, temp=CODER_TEMP, max_tokens=3000)
        current_code = extract_code(new_output)
    
    return None # Failed after retries

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

def main():
    logger.info(f"🚀 Initializing Ultra-Robust Builder with {MODEL_ID}")
    
    # Load Model
    try:
        model, tokenizer = load(MODEL_ID)
    except Exception as e:
        logger.critical(f"Model Load Failed: {e}")
        return

    # Stage 1: Plan
    stage_1_architect(model, tokenizer)
    
    # Sort files
    sort_dependency_order()
    
    # Stage 2: Build with recursive review
    stage_2_builder_loop(model, tokenizer)
    
    logger.info(f"🏁 Project Completed. Files saved in: {CODE_DIR}")
    logger.info("To run the project, navigate to the folder and run 'npm install' or 'pip install' based on generated configs.")

if __name__ == "__main__":
    main()
