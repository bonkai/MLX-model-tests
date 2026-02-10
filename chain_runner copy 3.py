#!/usr/bin/env python3
"""
MLX-LM Advanced MVP Builder
---------------------------
Features:
- Profile Switching (Creative vs. Strict)
- Atomic Step Granularity
- Automated JSON Validation & Retry Loop
- Context Injection (Previous answers feed into next steps)
- Detailed Logging
"""

import os
import json
import logging
import re
from datetime import datetime
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

MODEL_ID = "huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-mlx-4bit"
MAX_RETRIES = 4

# Define your project inputs here
INPUTS = {
    "project_name": "NewsFeed",
    "app_idea": "A web app for finding customized news for people to easily customize their news feed.",
    "tech_stack": "PHP, HTML, JavaScript, Tailwind CSS, MySQL."
}

# =============================================================================
# 2. GENERATION PROFILES
# =============================================================================

PROFILES = {
    "creative": {
        "description": "High temperature for brainstorming features.",
        "temp": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "max_tokens": 1000
    },
    "strict": {
        "description": "Low temperature for Code, JSON, and Logic.",
        "temp": 0.1, # Near deterministic
        "top_p": 0.85,
        "repetition_penalty": 1.1,
        "max_tokens": 1500
    }
}

# =============================================================================
# 3. THE ATOMIC STEP CHAIN
# =============================================================================

STEPS = [
    # --- STEP 1: Feature Refinement (Creative) ---
    {
        "step_id": "1_mvp_features",
        "profile": "creative",
        "role": "Product Manager",
        "description": "Define the MVP Core Features",
        "system_prompt": "You are a pragmatic Product Manager. You strip ideas down to the bare essentials.",
        "user_template": """
Project: {project_name}
Idea: {app_idea}

We need to define the MVP (Minimum Viable Product).
Return a JSON object with strictly this format:
{{
  "target_audience": "string",
  "core_value_prop": "string",
  "must_have_features": ["string", "string", "string"]
}}
Do not include nice-to-haves. Keep it lean.
""",
        "validation": "json",
        "required_keys": ["must_have_features", "core_value_prop"]
    },

    # --- STEP 2: Sitemap / User Flow (Creative/Balanced) ---
    {
        "step_id": "2_app_pages",
        "profile": "creative",
        "role": "UX Designer",
        "description": "Define the Page Structure",
        "system_prompt": "You are a UX Designer. Plan the site structure.",
        "user_template": """
Based on these features: {1_mvp_features[must_have_features]}

List the web pages required for the MVP.
Return a JSON object:
{{
  "pages": [
    {{ "path": "/", "name": "Landing", "purpose": "..." }},
    {{ "path": "/dashboard", "name": "Dashboard", "purpose": "..." }}
  ]
}}
""",
        "validation": "json",
        "required_keys": ["pages"]
    },

    # --- STEP 3: Database Entities (Strict) ---
    {
        "step_id": "3_db_tables_list",
        "profile": "strict",
        "role": "Database Architect",
        "description": "Identify Database Tables",
        "system_prompt": "You are a Senior DBA. Identify the necessary database entities.",
        "user_template": """
App: {project_name}
Stack: {tech_stack}
Features: {1_mvp_features[must_have_features]}

List the database table names required. Use snake_case.
Return JSON:
{{
  "table_names": ["users", "skills", ...]
}}
""",
        "validation": "json",
        "required_keys": ["table_names"]
    },

    # --- STEP 4: Schema Definition (Strict) ---
    {
        "step_id": "4_sql_schema",
        "profile": "strict",
        "role": "Database Architect",
        "description": "Write the SQL Schema",
        "system_prompt": "You are a MySQL Expert. Write valid SQL.",
        "user_template": """
Create the SQL schema for these tables: {3_db_tables_list[table_names]}.
Tech Stack: {tech_stack} (MySQL).

Rules:
1. Include `id` (uuid default gen_random_uuid()), `created_at` for ALL tables.
2. Define Foreign Keys explicitly.
3. Output ONLY the SQL code. No markdown wrapper if possible, or wrap in ```sql.
""",
        "validation": "text_exists", # We just check if it generated text, hard to validate SQL via JSON
        "validation_keyword": "CREATE TABLE" # Simple check to ensure it wrote SQL
    },

    # --- STEP 5: Directory Structure (Strict) ---
    {
        "step_id": "5_file_structure",
        "profile": "strict",
        "role": "System Architect",
        "description": "Generate File Tree",
        "system_prompt": "You are a Senior Architect. Output a text file tree.",
        "user_template": """
Stack: {tech_stack}
Pages: {2_app_pages[pages]}

Generate the directory structure for this project.
Use a text-tree format.
""",
        "validation": "text_exists",
        "validation_keyword": "app/" # Ensure it mentions the app folder
    }
]

# =============================================================================
# 4. UTILITY FUNCTIONS
# =============================================================================

def setup_logging():
    # Generate timestamped log file name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"mvp_generation_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger("MVP_Builder")
    logger.info(f"Logging to: {log_file}")
    return logger

def extract_json_block(text):
    """Robust extraction of JSON from chatty LLM output."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match: return match.group(1)
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match: return match.group(1)
    return text

def validate_output(text, step_config):
    """Decides which validation strategy to use."""
    v_type = step_config.get("validation", "none")

    if v_type == "json":
        json_str = extract_json_block(text)
        try:
            data = json.loads(json_str)
            # Check required keys
            req_keys = step_config.get("required_keys", [])
            missing = [k for k in req_keys if k not in data]
            if missing:
                return False, None, f"JSON missing keys: {missing}"
            return True, data, None
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON syntax: {str(e)}"

    elif v_type == "text_exists":
        keyword = step_config.get("validation_keyword")
        if keyword and keyword.lower() not in text.lower():
            return False, text, f"Output missing keyword: '{keyword}'"
        return True, text, None

    return True, text, None

def resolve_template(template_str, global_inputs, memory):
    """
    Replaces {placeholders} with actual data.
    Supports basic access like {1_mvp_features[must_have_features]}
    """
    # 1. First, replace global inputs
    for k, v in global_inputs.items():
        template_str = template_str.replace(f"{{{k}}}", str(v))
    
    # 2. Replace memory items (previous steps)
    # This is a simple regex replacer for {step_id[key]} pattern
    pattern = re.compile(r"\{(\w+)\[(\w+)\]\}")
    
    def replacer(match):
        step_key = match.group(1)
        dict_key = match.group(2)
        if step_key in memory and isinstance(memory[step_key], dict):
            return str(memory[step_key].get(dict_key, "UNKNOWN"))
        return match.group(0) # Return original if not found

    return pattern.sub(replacer, template_str)

# =============================================================================
# 5. MAIN ENGINE
# =============================================================================

def main():
    logger = setup_logging()
    logger.info(f"🚀 Starting MVP Builder using {MODEL_ID}")

    # Load Model
    try:
        model, tokenizer = load(MODEL_ID)
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        return

    # Setup Output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"mvp_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    
    memory = {} # Stores valid outputs

    for step in STEPS:
        step_id = step["step_id"]
        logger.info(f"--- Processing Step: {step['description']} ({step['profile']}) ---")
        
        # 1. Prepare Prompt
        user_prompt = resolve_template(step["user_template"], INPUTS, memory)
        sys_prompt = step["system_prompt"]
        
        # 2. Get Profile Settings
        prof = PROFILES[step["profile"]]
        sampler = make_sampler(temp=prof["temp"], top_p=prof["top_p"])
        logits_processors = make_logits_processors(repetition_penalty=prof["repetition_penalty"])

        # 3. Generation Loop (with Retries)
        success = False
        attempts = 0
        current_error = None
        
        while attempts < MAX_RETRIES and not success:
            attempts += 1
            
            # Prepare messages
            # If retrying, we append the error to the user prompt to nudge the LLM
            current_user_prompt = user_prompt
            if current_error:
                current_user_prompt += f"\n\nSYSTEM ALERT: Your previous output was invalid. \nError: {current_error}\nPlease fix the format and try again."

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": current_user_prompt}
            ]
            
            prompt_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            logger.info(f"   generating (Attempt {attempts})...")
            
            text_out = generate(
                model, tokenizer, 
                prompt=prompt_input, 
                max_tokens=prof["max_tokens"], 
                sampler=sampler,
                logits_processors=logits_processors,
                verbose=False
            )
            text_out = text_out.strip()

            # 4. Validate
            is_valid, clean_data, error_msg = validate_output(text_out, step)
            
            if is_valid:
                logger.info("   ✅ Validation Passed")
                memory[step_id] = clean_data
                success = True
                
                # Save to file
                ext = "json" if step["validation"] == "json" else "md"
                fname = f"{step_id}.{ext}"
                with open(os.path.join(out_dir, fname), "w") as f:
                    if ext == "json":
                        json.dump(clean_data, f, indent=2)
                    else:
                        f.write(text_out)
            else:
                logger.warning(f"   ❌ Validation Failed: {error_msg}")
                current_error = error_msg

        if not success:
            logger.error(f"🛑 Step {step_id} failed completely. Stopping chain.")
            break
            
    logger.info(f"🏁 Finished. Output directory: {out_dir}")

if __name__ == "__main__":
    main()