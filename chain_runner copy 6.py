#!/usr/bin/env python3
"""
MLX-LM Deep Debug Developer (Batch Edition - Crash Proof)
---------------------------------------------------------
1. LOADS MODEL ONCE.
2. ITERATES through 'ideas.json'.
3. IGNORES directories in file lists.
4. CATCHES errors per project and continues.
"""

import os
import json
import logging
import re
import traceback
import sys
from datetime import datetime
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# =============================================================================
# 1. CONFIGURATION & GLOBALS
# =============================================================================

MODEL_ID = "huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-mlx-4bit"
IDEAS_FILE = "ideas.json"

# Settings for the Reasoning Engine
CONTEXT_CHAR_LIMIT = 48000     
GENERATION_MAX_TOKENS = 12288  
TEMP_REASONING = 0.05          
TOP_P_REASONING = 0.85
MAX_FILE_RETRIES = 3

# These will be updated dynamically per project
PROJECT_SETTINGS = {}
OUTPUT_DIR = ""
CODE_DIR = ""

# Logging paths (updated per project)
INFO_LOG = ""
ERROR_LOG = ""
RAW_LOG = ""

# Setup Main Logger Base
logger = logging.getLogger("AutoDev")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

# =============================================================================
# 2. STATE MANAGEMENT
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
        # We don't log here anymore to avoid cluttering the console in batch mode
        # logger will handle it in the build phase

    def get_smart_context(self):
        context_msg = "--- PROJECT FILE STATUS ---\n"
        for f in self.manifest:
            status = "[DONE]" if f['path'] in self.completed_paths else "[PENDING]"
            context_msg += f"{status} {f['path']}: {f['desc']}\n"
        
        # NEW LOGIC: "Pin" the Critical Files
        context_msg += "\n--- CRITICAL CONTEXT (Always Visible) ---\n"
        
        # 1. Find Database Schema
        for path, code in self.files.items():
            if "database.sql" in path or "schema.sql" in path:
                context_msg += f"\n/* --- SCHEMA: {path} --- */\n{code}\n"
        
        # 2. Find DB Connection (looks for 'db.php' or 'config.php')
        for path, code in self.files.items():
            if path.endswith("db.php") or "config" in path:
                context_msg += f"\n/* --- DB CONNECTION: {path} --- */\n{code}\n"

        # 3. Add the single most recent file for immediate flow
        last_file = list(self.files.items())[-1:]
        if last_file:
            path, code = last_file[0]
            context_msg += f"\n/* --- PREVIOUS FILE: {path} --- */\n{code}\n"
            
        if len(context_msg) > CONTEXT_CHAR_LIMIT:
            context_msg = "..." + context_msg[-CONTEXT_CHAR_LIMIT:] 

        return context_msg

# Global state placeholder
state = None

# =============================================================================
# 3. UTILITIES & LOGGING SETUP
# =============================================================================

def setup_project_logging(project_name):
    """Resets log handlers for the specific project folder."""
    global INFO_LOG, ERROR_LOG, RAW_LOG, logger
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # New folder for this specific project
    base_output = os.path.join("outputs", f"{project_name}_{timestamp}")
    
    global OUTPUT_DIR, CODE_DIR
    OUTPUT_DIR = base_output
    CODE_DIR = os.path.join(OUTPUT_DIR, "src")
    os.makedirs(CODE_DIR, exist_ok=True)

    INFO_LOG = os.path.join(OUTPUT_DIR, "run_info.log")
    ERROR_LOG = os.path.join(OUTPUT_DIR, "run_errors.log")
    RAW_LOG = os.path.join(OUTPUT_DIR, "raw_thought_process.log")

    # Remove old handlers to prevent writing to previous project logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Re-add handlers
    h_info = logging.FileHandler(INFO_LOG, mode='w')
    h_info.setFormatter(formatter)
    h_info.setLevel(logging.INFO)

    h_err = logging.FileHandler(ERROR_LOG, mode='w')
    h_err.setFormatter(formatter)
    h_err.setLevel(logging.WARNING)

    h_con = logging.StreamHandler()
    h_con.setFormatter(formatter)
    h_con.setLevel(logging.INFO)

    logger.addHandler(h_info)
    logger.addHandler(h_err)
    logger.addHandler(h_con)
    
    return base_output

def log_raw_thought(tag, content):
    try:
        with open(RAW_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n[{datetime.now()}] TYPE: {tag}\n{'='*80}\n{content}\n")
    except Exception:
        pass # If logging fails, just ignore it to keep running

def strip_thoughts(text):
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()

def clean_code_block(text):
    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if match: return match.group(1).strip()
    return strip_thoughts(text)

def run_llm_deep_thought(model, tokenizer, sys_prompt, user_prompt):
    enhanced_sys = sys_prompt + """
\n[DEEP REASONING PROTOCOL ENABLED]
1. BEFORE writing any code or JSON, you MUST think step-by-step.
2. Enclose your reasoning inside <thinking>...</thinking> XML tags.
3. Analyze the dependencies, potential bugs, and logic flow.
4. Only after the <thinking> block is closed, provide the final output.
    """
    enhanced_sys += f"\nSTACK: {PROJECT_SETTINGS['stack']}\nMODE: OFFLINE GENERATION (No DB Connect)."

    messages = [{"role": "system", "content": enhanced_sys}, {"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    if len(prompt) > (CONTEXT_CHAR_LIMIT * 1.5): 
        prompt = prompt[-int(CONTEXT_CHAR_LIMIT * 1.5):]

    log_raw_thought("PROMPT_INPUT", prompt)
    
    sampler = make_sampler(temp=TEMP_REASONING, top_p=TOP_P_REASONING)
    
    try:
        output = generate(model, tokenizer, prompt=prompt, max_tokens=GENERATION_MAX_TOKENS, sampler=sampler, verbose=False)
        log_raw_thought("RAW_OUTPUT_WITH_THOUGHTS", output)
        return output.strip()
    except Exception as e:
        logger.error(f"❌ LLM Error: {e}")
        return ""

# =============================================================================
# 4. CORE STAGES
# =============================================================================

def stage_1_architect(model, tokenizer):
    logger.info("---------------------------------------------------")
    logger.info(f"🏗️  STAGE 1: ARCHITECTURE ({PROJECT_SETTINGS['name']})")
    logger.info("---------------------------------------------------")
    
    sys_p = "You are a CTO & Lead Architect."
    
    # CHANGED: Aggressive prompting for structure
    user = f"""
PROJECT: {PROJECT_SETTINGS['name']}
IDEA: {PROJECT_SETTINGS['idea']}
STACK: {PROJECT_SETTINGS['stack']}

Task:
1. Analyze the requirements deeply.
2. Design a ROBUST, MODULAR file structure.
3. You MUST include these directories/files:
   - includes/db.php (Class-based Database connection)
   - includes/functions.php (Helper functions)
   - includes/header.php & includes/footer.php (UI modularity)
   - api/ (JSON endpoints for AJAX)
   - assets/js/ (Logic files)
   - assets/css/ (Custom styles)
   - database.sql (Complete Schema)
   - database_seeds.sql (INSERT statements: 5 users, 10 items, dummy data to make app look alive)
   
4. TARGET: Plan for 10-15 files minimum.
5. NO generic "script.php". Use specific names (e.g., auth_login.php, dashboard.php).

Output Format:
<thinking>
... detailed analysis ...
</thinking>
```json
{{ "files": [ {{ "path": "includes/db.php", "desc": "PDO Singleton Class" }}, {{ "path": "api/fetch_data.php", "desc": "JSON Endpoint" }} ] }}
```
"""
    raw = run_llm_deep_thought(model, tokenizer, sys_p, user)
    
    try:
        json_match = re.search(r"(\{.*\})", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            valid_files = []
            for f in data['files']:
                # CRASH FIX 1: Ignore folders (trailing slash) or no extension
                if f['path'].endswith('/') or "." not in f['path']:
                    logger.warning(f"⚠️ Skipping invalid file/folder: {f['path']}")
                    continue

                if not any(f['path'].endswith(ext) for ext in PROJECT_SETTINGS['forbidden_extensions']):
                    if f['path'] not in PROJECT_SETTINGS['forbidden_files']:
                        valid_files.append(f)
            
            valid_files.sort(key=lambda x: 0 if "sql" in x['path'] else (1 if "config" in x['path'] else 5))
            state.manifest = valid_files
            logger.info(f"✅ Architecture Locked: {len(state.manifest)} files planned.")
        else:
            logger.error("❌ Failed to parse JSON plan.")
    except Exception as e:
        logger.error(f"❌ Architect Error: {e}")

def stage_2_smart_build(model, tokenizer):
    logger.info("---------------------------------------------------")
    logger.info(f"🔨 STAGE 2: BUILD EXECUTION ({PROJECT_SETTINGS['name']})")
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
    raw_response = generate_draft(model, tokenizer, path, desc)
    if not raw_response: return False
    final_code = clean_code_block(raw_response)

    for attempt in range(1, MAX_FILE_RETRIES + 1):
        review = review_code(model, tokenizer, path, final_code)
        
        if review['status'] == 'PASS':
            full_path = os.path.join(CODE_DIR, path)
            
            # CRASH FIX 2: Handle directory creation and file writing safely
            try:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(final_code)
                state.add_file(path, final_code)
                logger.info(f"   ✅ Saved: {path}")
                return True
            except IsADirectoryError:
                logger.error(f"   ⛔ CRITICAL: '{path}' is a directory, not a file. Skipping.")
                return False
            except Exception as e:
                logger.error(f"   ⛔ Write Error: {e}")
                return False
        
        logger.warning(f"   🚩 Fix Required (Attempt {attempt}): {review['issues']}")
        raw_response = generate_fix(model, tokenizer, path, final_code, review['issues'])
        final_code = clean_code_block(raw_response)

    return False 

def generate_draft(model, tokenizer, path, desc):
    sys_p = "You are a Senior Full-Stack Engineer."
    context = state.get_smart_context()
    
    # CHANGED: Added context about professional standards
    extra = ""
    if "sql" in path: extra = "Ensure VALID SQL. Use InnoDB. Add Indexes."
    if "seeds.sql" in path: extra = "Generate REALISTIC dummy data. Passwords should be '123456' (hashed with bcrypt if needed). Create Admin and Standard users."
    if "config" in path: extra = "Use generic placeholders. No DB connections."
    if ".js" in path: extra = "Use 'fetch' for API calls. No jQuery."
    if ".php" in path: extra = "Use try/catch blocks. STRICT TYPES. Modularize HTML (include header/footer)."

    prompt = f"""
TASK: Write file '{path}'
DESC: {desc}
DESIGN_SYSTEM: {PROJECT_SETTINGS.get('design_system', '')}
CONTEXT: {context}

INSTRUCTIONS:
1. Use <thinking> tags to plan the logic.
2. {extra}
3. Make it "Beefy": Handle edge cases, add comments, and ensure it looks like a production file, not a tutorial.
4. Output the final code in a ```php (or appropriate lang) block.
"""
    return run_llm_deep_thought(model, tokenizer, sys_p, prompt)

def review_code(model, tokenizer, path, clean_code):
    sys_p = "You are a Strict Code Auditor."
    display_code = clean_code
    if len(display_code) > 25000:
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
    raw = run_llm_deep_thought(model, tokenizer, sys_p, prompt)
    clean_result = strip_thoughts(raw)
    
    if "PASS" in clean_result: return {'status': 'PASS', 'issues': None}
    return {'status': 'FAIL', 'issues': clean_result}

def generate_fix(model, tokenizer, path, code, issues):
    sys_p = "You are a Debugger. Fix the code."
    prompt = f"""
FILE: {path}
ISSUES: {issues}
BROKEN CODE:
{code}

INSTRUCTIONS:
1. Use <thinking> to determine the fix.
2. Output the fully corrected code block.
"""
    return run_llm_deep_thought(model, tokenizer, sys_p, prompt)

# =============================================================================
# 5. MAIN LOOP
# =============================================================================

def process_projects(model, tokenizer):
    try:
        with open(IDEAS_FILE, 'r') as f:
            project_list = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: '{IDEAS_FILE}' not found. Please create it first.")
        return

    print(f"📦 Loaded {len(project_list)} projects from {IDEAS_FILE}")

    for idx, inputs in enumerate(project_list):
        # CRASH FIX 3: Wrap the entire project in a try/except block
        # If one project explodes, we log it and move to the next.
        try:
            # 1. SETUP PROJECT GLOBALS
            global PROJECT_SETTINGS, state
            
            PROJECT_SETTINGS = {
                "name": inputs["project_name"],
                "idea": inputs["app_idea"],
                # CHANGED: Added Object-Oriented, MVC, and API emphasis
                "stack": "Object-Oriented PHP 8 (MVC Pattern), HTML5, Modern JavaScript (ES6 Modules), Tailwind CSS (CDN), MySQL. Architecture: Separation of Concerns (Views, Logic, Database).",
                "design_system": "Color Palette: Indigo & Slate (Modern SaaS). Rounded corners: lg. Font: Inter/sans-serif. Shadows: Soft.",
                "forbidden_extensions": [".tsx", ".jsx", ".ts", ".py", ".rb", ".go", ".java", ".sh", ".bat"],
                "forbidden_files": ["install.php", "setup.php", "migration.php", "init_db.php"]
            }

            # 2. SETUP LOGGING & DIRECTORIES
            out_path = setup_project_logging(PROJECT_SETTINGS['name'])
            
            # 3. RESET STATE
            state = ProjectState()

            # 4. RUN
            print(f"\n🚀 STARTING PROJECT {idx+1}/{len(project_list)}: {PROJECT_SETTINGS['name']}")
            print(f"   📁 Output: {out_path}")
            
            stage_1_architect(model, tokenizer)
            stage_2_smart_build(model, tokenizer)
            
            logger.info(f"🏁 PROJECT COMPLETE: {PROJECT_SETTINGS['name']}")

        except Exception as e:
            # Catch ANY unexpected error in this project and keep going
            error_msg = f"🔥 CRASHED ON PROJECT {inputs['project_name']}: {e}"
            print(error_msg)
            # Log to the project specific log if available
            if logger: logger.critical(error_msg)
            # Log to general traceback to stderr
            traceback.print_exc()
            print("➡️ Skipping to next project...")
            continue

if __name__ == "__main__":
    try:
        print("🔌 Loading Model (This happens only once)...")
        model, tokenizer = load(MODEL_ID)
        process_projects(model, tokenizer)
        print("\n✨ ALL PROJECTS GENERATED (Failures were skipped) ✨")
    except Exception as e:
        print(f"GLOBAL CRITICAL CRASH: {e}")
        traceback.print_exc()
