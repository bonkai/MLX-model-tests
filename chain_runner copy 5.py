#!/usr/bin/env python3
"""
MLX-LM Deep Debug Developer (Deep Reasoning Edition)
----------------------------------------------------
1. DEEP THINKING: Forces the model to use <thinking> tags for Chain-of-Thought.
2. DUAL LOGGING: Separates "Story" (Info) from "Problems" (Errors).
3. OFFLINE SAFE: Generates SQL schemas but executes NOTHING.
4. SCAFFOLDED PROMPTS: Forces Plan -> Analysis -> Code structure.
"""

import os
import json
import logging
import re
import traceback
from datetime import datetime
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

MODEL_ID = "huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-mlx-4bit"

INPUTS = {
    "project_name": "NewsFeed",
    "app_idea": "A web app for finding customized news. Users select categories, curated feed, bookmark feature."
}

PROJECT_SETTINGS = {
    "name": INPUTS["project_name"],
    "idea": INPUTS["app_idea"],
    "stack": "PHP (Vanilla), HTML5, JavaScript (ES6), Tailwind CSS (CDN), MySQL.",
    "forbidden_extensions": [".tsx", ".jsx", ".ts", ".py", ".rb", ".go", ".java", ".sh", ".bat"],
    "forbidden_files": ["install.php", "setup.php", "migration.php", "init_db.php"]
}

# --- DEEP THINKING SETTINGS ---
# We give the model massive room to "think" before answering.
CONTEXT_CHAR_LIMIT = 48000     
GENERATION_MAX_TOKENS = 12288  # Increased to allow for <thinking> block + Code
TEMP_REASONING = 0.05          # Extremely low for pure logic
TOP_P_REASONING = 0.85

MAX_FILE_RETRIES = 3

# =============================================================================
# 2. LOGGING (SPLIT & VERBOSE)
# =============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join("outputs", f"newsfeed_deep_{timestamp}")
CODE_DIR = os.path.join(OUTPUT_DIR, "src")
os.makedirs(CODE_DIR, exist_ok=True)

INFO_LOG = os.path.join(OUTPUT_DIR, "run_info.log")
ERROR_LOG = os.path.join(OUTPUT_DIR, "run_errors.log")
RAW_LOG = os.path.join(OUTPUT_DIR, "raw_thought_process.log")

# Setup Main Logger
logger = logging.getLogger("AutoDev")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

# Handler 1: Info (The Story)
h_info = logging.FileHandler(INFO_LOG, mode='w')
h_info.setLevel(logging.INFO)
h_info.setFormatter(formatter)

# Handler 2: Errors (The Problems)
h_err = logging.FileHandler(ERROR_LOG, mode='w')
h_err.setLevel(logging.WARNING)
h_err.setFormatter(formatter)

# Handler 3: Console
h_con = logging.StreamHandler()
h_con.setLevel(logging.INFO)
h_con.setFormatter(formatter)

logger.addHandler(h_info)
logger.addHandler(h_err)
logger.addHandler(h_con)

def log_raw_thought(tag, content):
    """Logs the full Chain-of-Thought (Thinking tags + Code) to a separate file."""
    with open(RAW_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n[{datetime.now()}] TYPE: {tag}\n{'='*80}\n{content}\n")

logger.info(f"🚀 STARTING DEEP REASONING ENGINE (FIXED REVIEWER)")

# =============================================================================
# 3. STATE MANAGEMENT
# =============================================================================

class ProjectState:
    def __init__(self):
        self.manifest = []
        self.files = {} 
        self.failed_queue = []
        self.completed_paths = set()

    def add_file(self, path, code):
        self.completed_paths.add(path)
        self.files[path] = code
        logger.info(f"💾 File Committed: {path}")

    def get_smart_context(self):
        # Sliding Window Context
        context_msg = "--- PROJECT FILE STATUS ---\n"
        for f in self.manifest:
            status = "[DONE]" if f['path'] in self.completed_paths else "[PENDING]"
            context_msg += f"{status} {f['path']}: {f['desc']}\n"
        
        context_msg += "\n--- RECENT CODE (Reference Only) ---\n"
        # Only show last 2 files to keep prompt clean
        recent_files = list(self.files.items())[-2:] 
        
        for path, code in recent_files:
            entry = f"\n/* --- FILE: {path} --- */\n{code}\n/* --- END --- */\n"
            context_msg += entry
            
        if len(context_msg) > CONTEXT_CHAR_LIMIT:
            context_msg = "..." + context_msg[-CONTEXT_CHAR_LIMIT:] 

        return context_msg

state = ProjectState()

# =============================================================================
# 4. UTILITIES
# =============================================================================

def strip_thoughts(text):
    """
    Removes the <thinking>...</thinking> block completely.
    Used before sending code to the Reviewer to save token space.
    """
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()

def clean_code_block(text):
    """
    Extracts pure code from Markdown, ignoring thoughts.
    """
    # 1. Regex to find the Markdown Block (```php ... ```)
    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if match: return match.group(1).strip()
    
    # 2. Fallback: Strip thoughts and return what's left
    return strip_thoughts(text)

def run_llm_deep_thought(model, tokenizer, sys_prompt, user_prompt):
    """
    The engine that forces 'Deep Thinking' instructions.
    """
    # 1. Inject the Reasoning Framework into System Prompt
    enhanced_sys = sys_prompt + """
\n[DEEP REASONING PROTOCOL ENABLED]
1. BEFORE writing any code or JSON, you MUST think step-by-step.
2. Enclose your reasoning inside <thinking>...</thinking> XML tags.
3. Analyze the dependencies, potential bugs, and logic flow.
4. Only after the <thinking> block is closed, provide the final output.
    """

    # 2. Inject Offline Constraints
    enhanced_sys += f"\nSTACK: {PROJECT_SETTINGS['stack']}\nMODE: OFFLINE GENERATION (No DB Connect)."

    messages = [{"role": "system", "content": enhanced_sys}, {"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 3. Safety Truncation
    if len(prompt) > (CONTEXT_CHAR_LIMIT * 1.5): 
        logger.warning(f"⚠️ Input Prompt too long ({len(prompt)}). Truncating history.")
        prompt = prompt[-int(CONTEXT_CHAR_LIMIT * 1.5):]

    log_raw_thought("PROMPT_INPUT", prompt)
    logger.info(f"🤖 Thinking... (Prompt Length: {len(prompt)} chars)")
    
    # 4. Run Generation with High Precision (Low Temp)
    sampler = make_sampler(temp=TEMP_REASONING, top_p=TOP_P_REASONING)
    
    try:
        output = generate(model, tokenizer, prompt=prompt, max_tokens=GENERATION_MAX_TOKENS, sampler=sampler, verbose=False)
        log_raw_thought("RAW_OUTPUT_WITH_THOUGHTS", output)
        
        # Log just the "Thinking" part to the console for the user to see it's working
        think_match = re.search(r"<thinking>(.*?)</thinking>", output, re.DOTALL)
        if think_match:
            thought_snippet = think_match.group(1)[:200].replace("\n", " ")
            logger.info(f"   💭 Thoughts: {thought_snippet}...")
        else:
            logger.warning("   ⚠️ Model skipped explicit <thinking> tags.")
            
        return output.strip()
    except Exception as e:
        logger.error(f"❌ LLM Error: {e}")
        logger.error(traceback.format_exc())
        return ""

# =============================================================================
# 5. CORE STAGES
# =============================================================================

def stage_1_architect(model, tokenizer):
    logger.info("---------------------------------------------------")
    logger.info("🏗️  STAGE 1: ARCHITECTURE (Deep Plan)")
    logger.info("---------------------------------------------------")
    
    sys = "You are a Senior Software Architect."
    user = f"""
PROJECT: {PROJECT_SETTINGS['name']}
IDEA: {PROJECT_SETTINGS['idea']}

Task:
1. Analyze the requirements deeply inside <thinking> tags.
2. Output a JSON object containing the file structure.
3. Include 'database.sql' (Priority 0).
4. Include 'config.php' (Priority 1).
5. NO install/setup scripts.

Output Format:
<thinking>
... analysis ...
</thinking>
```json
{{ "files": [ {{ "path": "index.php", "desc": "Entry point" }} ] }}
```
"""
    raw = run_llm_deep_thought(model, tokenizer, sys, user)
    
    try:
        json_match = re.search(r"(\{.*\})", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            
            valid_files = []
            for f in data['files']:
                if not any(f['path'].endswith(ext) for ext in PROJECT_SETTINGS['forbidden_extensions']):
                    if f['path'] not in PROJECT_SETTINGS['forbidden_files']:
                        valid_files.append(f)
            
            # Sort: SQL -> Config -> Others
            valid_files.sort(key=lambda x: 0 if "sql" in x['path'] else (1 if "config" in x['path'] else 5))
            state.manifest = valid_files
            logger.info(f"✅ Architecture Locked: {len(state.manifest)} files planned.")
        else:
            logger.error("❌ Failed to parse JSON plan.")
    except Exception as e:
        logger.error(f"❌ Architect Error: {e}")

def stage_2_smart_build(model, tokenizer):
    logger.info("---------------------------------------------------")
    logger.info("🔨 STAGE 2: DEEP BUILD EXECUTION")
    logger.info("---------------------------------------------------")
    
    for i, file_info in enumerate(state.manifest):
        path = file_info['path']
        logger.info(f"👉 Building [{i+1}/{len(state.manifest)}]: {path}")
        
        success = build_file_with_reasoning(model, tokenizer, path, file_info['desc'])
        if not success:
            state.failed_queue.append(file_info)

    if state.failed_queue:
        logger.info(f"🧹 Retrying {len(state.failed_queue)} failed files...")
        for file_info in state.failed_queue:
            build_file_with_reasoning(model, tokenizer, file_info['path'], file_info['desc'])

def build_file_with_reasoning(model, tokenizer, path, desc):
    """
    Generates code using a 'Plan -> Code' flow.
    """
    # 1. Draft
    # We get the RAW response (including thoughts)
    raw_response = generate_draft(model, tokenizer, path, desc)
    if not raw_response: return False
    
    # We extract the pure code for saving later
    final_code = clean_code_block(raw_response)

    # 2. Audit & Fix Loop
    for attempt in range(1, MAX_FILE_RETRIES + 1):
        # FIX: We pass the 'final_code' (stripped of thoughts), NOT the raw_response
        review = review_code(model, tokenizer, path, final_code)
        
        if review['status'] == 'PASS':
            full_path = os.path.join(CODE_DIR, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(final_code)
            state.add_file(path, final_code)
            logger.info(f"   ✅ Saved: {path}")
            return True
        
        logger.warning(f"   🚩 Fix Required (Attempt {attempt}): {review['issues']}")
        
        # When fixing, we usually feed the code back. 
        # Pass the CLEAN code so the fixer sees just the PHP, not the previous thinking.
        raw_response = generate_fix(model, tokenizer, path, final_code, review['issues'])
        final_code = clean_code_block(raw_response)

    return False 

def generate_draft(model, tokenizer, path, desc):
    sys = "You are an Expert PHP Developer."
    context = state.get_smart_context()
    
    extra = ""
    if "sql" in path: extra = "Ensure VALID SQL. No dummy data."
    if "config" in path: extra = "Use generic placeholders. No DB connections."

    prompt = f"""
TASK: Write file '{path}'
DESC: {desc}
CONTEXT: {context}

INSTRUCTIONS:
1. Use <thinking> tags to plan the code.
2. {extra}
3. Output the final code in a ```php (or sql) block.
"""
    # Returns RAW string (Thinking + Code)
    return run_llm_deep_thought(model, tokenizer, sys, prompt)

def review_code(model, tokenizer, path, clean_code):
    """
    CRITICAL FIX: 
    1. Input `clean_code` (thoughts already removed).
    2. NO slicing ([:5000] removed).
    """
    sys = "You are a Strict Code Auditor."
    
    # Safety check: If the CODE is larger than context, we must slice, 
    # but let's make it 25k chars, not 5k.
    display_code = clean_code
    if len(display_code) > 25000:
        logger.warning("   ⚠️ File extremely large (>25k chars). Truncating for review.")
        display_code = display_code[:25000] + "\n...[TRUNCATED]..."

    prompt = f"""
Review this code for '{path}'.

CODE:
{display_code}

INSTRUCTIONS:
1. Use <thinking> tags to analyze.
2. Check for Syntax Errors, Missing Closing Tags, or Forbidden Commands.
3. Output ONLY: "PASS" or "FAIL: <reason>".
"""
    raw = run_llm_deep_thought(model, tokenizer, sys, prompt)
    clean_result = strip_thoughts(raw)
    
    if "PASS" in clean_result: return {'status': 'PASS', 'issues': None}
    return {'status': 'FAIL', 'issues': clean_result}

def generate_fix(model, tokenizer, path, code, issues):
    sys = "You are a Debugger. Fix the code."
    prompt = f"""
FILE: {path}
ISSUES: {issues}
BROKEN CODE:
{code}

INSTRUCTIONS:
1. Use <thinking> to determine the fix.
2. Output the fully corrected code block.
"""
    return run_llm_deep_thought(model, tokenizer, sys, prompt)

# =============================================================================
# 6. RUN
# =============================================================================

if __name__ == "__main__":
    try:
        logger.info("🔌 Loading Model...")
        model, tokenizer = load(MODEL_ID)
        stage_1_architect(model, tokenizer)
        stage_2_smart_build(model, tokenizer)
        logger.info("🏁 DONE. Check 'raw_thought_process.log' to see the reasoning.")
    except Exception as e:
        logger.critical(f"CRASH: {e}")
        logger.critical(traceback.format_exc())
