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

# Platform-aware helpers / protocol
def slugify(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "app"

VENTURE_STUDIO_PROTOCOL = r"""
### SYSTEM CONTEXT: The Venture Studio Protocol

The Mission:
We are building a Venture Studio—a rapid-development ecosystem designed to launch independent SaaS applications (20+) using a shared "Monolith" architecture. The goal is extreme speed: we share Authentication, Billing, User Management, and Infrastructure so that building a new app only requires building the unique logic for that specific tool.

Core Tech Stack (Strict Constraints):
- Server: Nginx + PHP 8.x (Native/Vanilla. NO Laravel/Symfony/CodeIgniter).
- Database: MySQL (Raw PDO queries).
- Frontend: HTML5, CSS (Tailwind), Vanilla JavaScript.
- No Build Steps: We do NOT use npm, webpack, or React. We use raw script tags and CSS files.

The Architecture ("The Magic Router"):
We use a single Nginx entry point. All subdomains (*.societymediatech.com) point to a single public/index.php.
- Logic: The PHP script detects the subdomain (e.g., mail, todo, crm), looks up the app in the apps database table, and routes the request to the specific folder in /apps/.
- Isolation: To the user, each subdomain feels like a completely separate product. To us, it is one codebase.

Current Directory Structure:
```text
/new_platform
├── /apps                  <-- ISOLATED APP LOGIC (The "Venture" layer)
│   ├── /studio            <-- Admin Dashboard (manages global users/revenue)
│   ├── /mail              <-- Example App: "VentureMail"
│   └── /template          <-- The Blueprint for new apps
│       ├── views/         <-- App-specific HTML/Tailwind views
│       └── routes.php     <-- App-specific routing logic
│
├── /shared                <-- THE CORE (Do not modify unless necessary)
│   ├── /auth              <-- Login, Register, Google OAuth logic
│   ├── /views             <-- Global Partials (Nav, Footer, Alerts) & Legal pages
│   ├── /assets            <-- Shared CSS (Tailwind), JS helpers, and Logos
│   ├── Auth.php           <-- Class: Handles Sessions, Memberships, & Access Control
│   ├── Database.php       <-- Class: PDO Singleton
│   └── config.php         <-- ENV variables, DB Creds, API Keys
│
└── /public
    └── index.php          <-- THE CONTROLLER. Handles Routing, Subdomain detection, & Loading.
```

Database Strategy (MySQL):
We have two types of data:
1) Global Tables (Existing):
- users
- apps -> The registry of valid subdomains.
- app_memberships -> Controls access. If a user isn't in this table for the current subdomain, they get redirected to pricing.
2) App-Specific Tables (To Be Created):
- Every new app needs its own tables (e.g., mail_messages, todo_tasks).
- Rule: Always prefix app-specific tables with the app slug to avoid collisions (e.g., crm_customers, crm_leads).

Critical Operational Constraints:
1) I am the Hands, You are the Brain: You have full access to Create, Edit, and Delete files in new_platform/. These auto-upload to the live server.
2) NO Server/Shell Access: You cannot run php, composer, mysql, or npm commands. You cannot restart the server.
3) Command Protocol: If a database change is needed, or a dependency is required, you must provide the exact SQL query to run manually.
4) No Local Testing: Do not ask to curl localhost.

THE TASK: Initialize New App

### PLATFORM DIRECTORY CONVENTIONS (IMPORTANT)
- App code MUST be generated under: /apps/<slug>/
- Static assets MUST be generated under: /public/assets/<slug>/
- Shared core exists already under: /shared (Auth.php, Database.php, config.php)
- DO NOT create or modify /shared files.
- DO NOT create standalone includes/db.php, includes/header.php, includes/footer.php, etc.
- public/index.php is the bootstrap; app routes/views SHOULD NOT re-require shared files.
- In apps/<slug>/routes.php, you MAY have app-level public routes (e.g., /embed/<id>) handled BEFORE Auth::requireAccess(); then gate the rest.
- Use raw PDO via Database::connect() (shared/Database.php).
- Use Auth::user() + Auth::requireAccess() for login + membership gating.
- Do NOT create app-level /login, /register, /logout, /pricing, /privacy, /terms routes (global router handles them).
- Tailwind via CDN only. No npm/build tools.
"""

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

# Optional: detect real platform code for in-context reference (read-only)
PLATFORM_ROOT = None
PLATFORM_REFERENCE = ""

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
        
        # 1. Find Database Schema / Migrations
        for path, code in self.files.items():
            if path.endswith(".sql") or "/migrations/" in path:
                context_msg += f"\n/* --- SQL: {path} --- */\n{code}\n"
        
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
    base_output = os.path.join("outputs", f"{slugify(project_name)}_{timestamp}")
    
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

def detect_platform_root():
    """
    Best-effort detection of the Venture Studio platform root.
    We DO NOT require this to exist; the generator can still run without it.
    """
    candidates = []
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.extend([
            os.path.join(here, "..", "VENTURE_STUDIO", "new_platform"),
            os.path.join(here, "..", "new_platform"),
        ])
    except Exception:
        pass

    try:
        cwd = os.getcwd()
        candidates.extend([
            os.path.join(cwd, "VENTURE_STUDIO", "new_platform"),
            os.path.join(cwd, "new_platform"),
        ])
    except Exception:
        pass

    for c in candidates:
        try:
            root = os.path.abspath(c)
            if (
                os.path.isfile(os.path.join(root, "public", "index.php"))
                and os.path.isfile(os.path.join(root, "shared", "Auth.php"))
                and os.path.isfile(os.path.join(root, "shared", "Database.php"))
            ):
                return root
        except Exception:
            continue

    return None

def safe_read_text(path, max_chars=None):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if isinstance(max_chars, int) and max_chars > 0 and len(content) > max_chars:
            return content[:max_chars] + "\n...[TRUNCATED]..."
        return content
    except Exception:
        return ""

def build_platform_reference(platform_root):
    """
    Pull in a small, safe snapshot of real platform files to reduce guesswork.
    NOTE: Do NOT include shared/config.php (can contain secrets).
    """
    if not platform_root or not os.path.isdir(platform_root):
        return ""

    ref_files = [
        ("public/index.php", 7000),
        ("shared/Auth.php", 6000),
        ("shared/Database.php", 3000),
        ("apps/todo/routes.php", 2500),
        ("apps/todo/views/home.php", 3500),
        ("apps/todo/views/dashboard.php", 3500),
        ("apps/marketing/routes.php", 2500),
        # Real production-style router pattern (public routes + gated routes)
        ("apps/vella/routes.php", 3400),
    ]

    parts = []
    for rel, max_chars in ref_files:
        abs_path = os.path.join(platform_root, rel)
        if not os.path.isfile(abs_path):
            continue
        code = safe_read_text(abs_path, max_chars=max_chars)
        if not code.strip():
            continue
        parts.append(f"\n/* --- PLATFORM REFERENCE (READ ONLY): {rel} --- */\n{code}\n")

    snapshot = "\n".join(parts).strip()
    if len(snapshot) > 24000:
        snapshot = snapshot[:24000] + "\n...[TRUNCATED]..."
    return snapshot

def safe_join(base_dir, relative_path):
    """
    Prevent path traversal / absolute path writes.
    Returns an absolute path that is guaranteed to be under base_dir.
    """
    base_abs = os.path.abspath(base_dir)
    target_abs = os.path.abspath(os.path.join(base_abs, relative_path))
    if not target_abs.startswith(base_abs + os.sep):
        raise ValueError(f"Unsafe output path: {relative_path}")
    return target_abs

def run_llm_deep_thought(model, tokenizer, sys_prompt, user_prompt):
    enhanced_sys = (
        VENTURE_STUDIO_PROTOCOL
        + "\n\n"
        + sys_prompt
        + """
\n[DEEP REASONING PROTOCOL ENABLED]
1. BEFORE writing any code or JSON, you MUST think step-by-step.
2. Enclose your reasoning inside <thinking>...</thinking> XML tags.
3. Analyze the dependencies, potential bugs, and logic flow.
4. Only after the <thinking> block is closed, provide the final output.
"""
    )
    enhanced_sys += f"\nAPP_NAME: {PROJECT_SETTINGS.get('name', '')}"
    enhanced_sys += f"\nAPP_SLUG: {PROJECT_SETTINGS.get('slug', '')}"
    enhanced_sys += "\nCONSTRAINTS: Vanilla PHP 8 + MySQL (PDO) + Tailwind CDN + Vanilla JS. No build steps. No frameworks."
    enhanced_sys += "\nNOTE: Shared core is at /shared. Apps live at /apps/<slug>. Assets live at /public/assets/<slug>."
    enhanced_sys += f"\nSTACK: {PROJECT_SETTINGS.get('stack', '')}\nMODE: OFFLINE GENERATION (No DB Connect)."
    if PLATFORM_REFERENCE:
        enhanced_sys += "\n\n### PLATFORM REFERENCE FILES (READ ONLY)\n" + PLATFORM_REFERENCE

    messages = [{"role": "system", "content": enhanced_sys}, {"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Prevent prompt overflow while preserving the SYSTEM instructions.
    max_len = int(CONTEXT_CHAR_LIMIT * 1.5)
    if len(prompt) > max_len:
        head = int(max_len * 0.35)
        tail = max_len - head
        prompt = prompt[:head] + "\n...[TRUNCATED MIDDLE]...\n" + prompt[-tail:]

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

def get_required_manifest_files(app_slug):
    return [
        {"path": f"apps/{app_slug}/routes.php", "desc": "App routes wiring into shared core (public/index.php includes shared)."},
        {"path": f"apps/{app_slug}/views/home.php", "desc": "Public landing page view."},
        {"path": f"apps/{app_slug}/views/dashboard.php", "desc": "Protected dashboard view (Auth::requireAccess)."},
        {"path": f"apps/{app_slug}/migrations/001_create_tables.sql", "desc": "SQL migration to create app tables (prefix all tables with slug_)."},
        {"path": f"apps/{app_slug}/migrations/002_seed.sql", "desc": "SQL seed file for app tables only (optional but useful)."},
        {"path": f"apps/{app_slug}/README.md", "desc": "Manual setup steps (routing + SQL to run + deployment checklist)."},
        {"path": f"public/assets/{app_slug}/app.js", "desc": "Vanilla JS for app interactions (no build tools)."},
        {"path": f"public/assets/{app_slug}/app.css", "desc": "App-specific CSS (optional; Tailwind via CDN)."},
    ]

def normalize_planned_path(raw_path, app_slug):
    if not isinstance(raw_path, str):
        return None
    p = raw_path.strip()
    if not p:
        return None

    p = p.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    p = re.sub(r"/+", "/", p)

    # If the model includes the repo root prefix, strip it.
    if p.startswith("VENTURE_STUDIO/new_platform/"):
        p = p[len("VENTURE_STUDIO/new_platform/"):]
    if p.startswith("new_platform/"):
        p = p[len("new_platform/"):]

    # Replace common placeholder patterns the model might emit.
    p = (
        p.replace("<slug>", app_slug)
        .replace("{slug}", app_slug)
        .replace("{PROJECT_SETTINGS['slug']}", app_slug)
        .replace('{PROJECT_SETTINGS["slug"]}', app_slug)
    )

    base = os.path.basename(p)
    base_lower = base.lower()

    # Legacy mini-app outputs -> map to Venture platform equivalents.
    if base_lower in ("database.sql", "schema.sql"):
        return f"apps/{app_slug}/migrations/001_create_tables.sql"
    if base_lower in ("database_seeds.sql", "database_seed.sql", "seeds.sql", "seed.sql"):
        return f"apps/{app_slug}/migrations/002_seed.sql"
    if p.startswith("assets/js/") and base_lower.endswith(".js"):
        return f"public/assets/{app_slug}/{base}"
    if p.startswith("assets/css/") and base_lower.endswith(".css"):
        return f"public/assets/{app_slug}/{base}"

    return p

def is_allowed_app_path(path, app_slug):
    return (
        isinstance(path, str)
        and (path.startswith(f"apps/{app_slug}/") or path.startswith(f"public/assets/{app_slug}/"))
    )

def normalize_manifest_entry(entry, app_slug):
    if not isinstance(entry, dict):
        return None

    raw_path = entry.get("path")
    raw_desc = entry.get("desc", "")
    path = normalize_planned_path(raw_path, app_slug)
    if not path:
        return None

    # Reject wildcard / template paths. We need concrete file paths to generate.
    if any(ch in path for ch in ["*", "?"]):
        return None
    if any(ch in path for ch in ["{", "}", "<", ">"]):
        return None

    # Must be a file, not a folder.
    if path.endswith("/") or "." not in os.path.basename(path):
        return None

    # Block path traversal.
    segments = [seg for seg in path.split("/") if seg]
    if any(seg == ".." for seg in segments):
        return None

    # Hard enforce platform output roots.
    if not is_allowed_app_path(path, app_slug):
        return None

    # Enforce forbidden extensions/files.
    forbidden_exts = PROJECT_SETTINGS.get("forbidden_extensions") or []
    if any(isinstance(ext, str) and path.endswith(ext) for ext in forbidden_exts):
        return None

    forbidden_files = PROJECT_SETTINGS.get("forbidden_files") or []
    base = os.path.basename(path)
    if base in forbidden_files or path in forbidden_files:
        return None

    desc = raw_desc if isinstance(raw_desc, str) else ""
    desc = desc.strip() or "Generated file"
    return {"path": path, "desc": desc}

def finalize_manifest(planned_files, app_slug):
    required = get_required_manifest_files(app_slug)
    required_paths = {f["path"] for f in required}

    out = []
    seen = set()

    # Start with required files no matter what.
    for f in required:
        out.append(f)
        seen.add(f["path"])

    # Then accept any additional normalized/valid files.
    if isinstance(planned_files, list):
        for f in planned_files:
            norm = normalize_manifest_entry(f, app_slug)
            if not norm:
                continue
            if norm["path"] in seen:
                continue
            out.append(norm)
            seen.add(norm["path"])

    # Sort into a sane build order (migrations -> routes -> PHP -> assets).
    def sort_key(item):
        path = item.get("path", "")
        if "/migrations/" in path or path.endswith(".sql"):
            pri = 0
        elif path.endswith("routes.php"):
            pri = 1
        elif "/controllers/" in path:
            pri = 2
        elif "/models/" in path:
            pri = 3
        elif "/views/" in path:
            pri = 4
        elif path.endswith(".css"):
            pri = 6
        elif path.endswith(".js"):
            pri = 7
        else:
            pri = 10
        return (pri, path)

    out.sort(key=sort_key)
    return out

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
2. Design a ROBUST, MODULAR file structure that fits THIS platform.

REQUIRED OUTPUT FILES (must be included):
- apps/{PROJECT_SETTINGS['slug']}/routes.php
- apps/{PROJECT_SETTINGS['slug']}/views/home.php
- apps/{PROJECT_SETTINGS['slug']}/views/dashboard.php
- apps/{PROJECT_SETTINGS['slug']}/migrations/001_create_tables.sql
- apps/{PROJECT_SETTINGS['slug']}/migrations/002_seed.sql
- apps/{PROJECT_SETTINGS['slug']}/README.md
- public/assets/{PROJECT_SETTINGS['slug']}/app.js
- public/assets/{PROJECT_SETTINGS['slug']}/app.css

OPTIONAL (include if needed by the app):
- apps/{PROJECT_SETTINGS['slug']}/controllers/*.php
- apps/{PROJECT_SETTINGS['slug']}/models/*.php
- apps/{PROJECT_SETTINGS['slug']}/views/*.php (other pages)
(If you include optional files, enumerate REAL file paths — no wildcards like *.php.)

RULES:
- DO NOT create /shared files (Auth.php, Database.php, config.php already exist).
- DO NOT create includes/db.php or any standalone DB connector.
- All app tables MUST be prefixed with "{PROJECT_SETTINGS['slug']}_".
- Views should load assets from "/assets/{PROJECT_SETTINGS['slug']}/app.js" and "/assets/{PROJECT_SETTINGS['slug']}/app.css".
- All DB access must use shared/Database.php (PDO).
- In JSON output, every entry must be a concrete file path (no '*' wildcards).

Output Format:
<thinking>
... detailed analysis ...
</thinking>
```json
{{ "files": [ {{ "path": "apps/{PROJECT_SETTINGS['slug']}/routes.php", "desc": "App routes wiring into shared core" }}, {{ "path": "apps/{PROJECT_SETTINGS['slug']}/views/home.php", "desc": "Public landing page view" }}, {{ "path": "public/assets/{PROJECT_SETTINGS['slug']}/app.js", "desc": "Vanilla JS for UI interactions" }} ] }}
```
"""
    raw = run_llm_deep_thought(model, tokenizer, sys_p, user)
    
    app_slug = PROJECT_SETTINGS.get("slug") or slugify(PROJECT_SETTINGS.get("name", "app"))
    PROJECT_SETTINGS["slug"] = app_slug

    planned_files = []
    parse_ok = False
    try:
        # Prefer the fenced ```json block if present.
        candidate = clean_code_block(raw)
        data = json.loads(candidate)
        if isinstance(data, dict) and isinstance(data.get("files"), list):
            planned_files = data["files"]
            parse_ok = True
    except Exception:
        pass

    if not parse_ok:
        try:
            # Fallback: find the first JSON object in the raw output.
            json_match = re.search(r"(\{[\s\S]*\})", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and isinstance(data.get("files"), list):
                    planned_files = data["files"]
                    parse_ok = True
        except Exception:
            parse_ok = False

    if not parse_ok:
        logger.error("❌ Failed to parse JSON plan. Falling back to required platform files only.")

    state.manifest = finalize_manifest(planned_files, app_slug)
    logger.info(f"✅ Architecture Locked: {len(state.manifest)} files planned.")

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
            try:
                full_path = safe_join(CODE_DIR, path)
            except Exception as e:
                logger.error(f"   ⛔ Unsafe path refused: {path} ({e})")
                return False
            
            # CRASH FIX 2: Handle directory creation and file writing safely
            try:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
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
    lower_path = (path or "").lower()
    app_slug = PROJECT_SETTINGS.get("slug", "app")

    if lower_path.endswith(".sql"):
        extra = (
            "Ensure VALID MySQL SQL. Use InnoDB. Add indexes. "
            f"All app tables MUST be prefixed with '{app_slug}_'. "
            "Do NOT create/modify global tables (users, apps, app_memberships)."
        )
    if "seed" in lower_path and lower_path.endswith(".sql"):
        extra = (
            f"Seed ONLY {app_slug}_* tables with realistic dummy data. "
            "Do NOT touch global tables. Keep it small (5-20 rows per table)."
        )
    if lower_path.endswith(".md"):
        extra = (
            "Write a concise Markdown README for THIS app. Include:\n"
            "- What the app does + its slug\n"
            "- Routes overview (public vs gated)\n"
            "- Manual platform wiring steps (IMPORTANT):\n"
            "  1) Add the subdomain->slug mapping in public/index.php ($appMap) so the subdomain routes to this app.\n"
            "  2) Insert the app into the global apps table so Auth::requireAccess() works.\n"
            "     Example:\n"
            f"     INSERT INTO apps (name, slug) VALUES ('{PROJECT_SETTINGS.get('name','App')}', '{app_slug}');\n"
            "     (If your apps table has additional required columns, include them.)\n"
            "- DB setup: run apps/<slug>/migrations/001_create_tables.sql then 002_seed.sql\n"
            "- Notes: do not modify /shared; global auth/legal routes are handled by public/index.php\n"
        )
    if lower_path.endswith(".js"):
        extra = "Vanilla JS only (no build tools, no modules). Use fetch() for API calls. No jQuery."
    if lower_path.endswith(".css"):
        extra = "Keep CSS minimal; Tailwind via CDN is primary. Only add truly app-specific tweaks."
    if lower_path.endswith(".php"):
        if lower_path.endswith("/routes.php"):
            extra = (
                "This is an app router (apps/<slug>/routes.php). "
                "public/index.php already bootstraps shared/config.php, shared/Database.php, shared/Auth.php and starts the session. "
                "Do NOT require shared files here. "
                "Parse the current path via parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH). "
                "If you need app-level PUBLIC routes (e.g., /embed/<id>), handle them BEFORE Auth::requireAccess(); then gate the rest. "
                "For gated routes, call Auth::requireAccess(); then require __DIR__ . '/views/<page>.php'. "
                "Do NOT implement global routes (/login, /register, /logout, /pricing, /privacy, /terms). "
                f"Any app tables must be prefixed '{app_slug}_'. Use Database::connect() and prepared statements."
            )
        elif "/views/" in lower_path:
            extra = (
                "This is an app view (apps/<slug>/views/*.php). "
                "Do NOT require shared files (public/index.php already bootstraps). "
                "Use Auth::user() if you need the current user (id/name/email). "
                f"Load app assets from /assets/{app_slug}/app.css and /assets/{app_slug}/app.js (defer). "
                "Load Tailwind via CDN: <script src=\"https://cdn.tailwindcss.com\"></script>. "
                "Escape dynamic output with htmlspecialchars()."
            )
        elif "/controllers/" in lower_path or "/models/" in lower_path:
            extra = (
                "This is app backend code (controllers/models). "
                "Do NOT require shared files (public/index.php already bootstraps). "
                "Use Database::connect() (PDO) and prepared statements. "
                f"Any app tables must be prefixed '{app_slug}_'. "
                "Avoid assuming global DB columns; rely on Auth::user() for user identity."
            )
        else:
            extra = (
                "Match Venture Studio platform conventions. "
                "Do NOT require shared files (public/index.php bootstraps). "
                "Use Database::connect() (PDO) and Auth::user()/Auth::requireAccess() as needed."
            )

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
    
    verdict = clean_result.strip()
    if verdict.startswith("PASS"):
        return {'status': 'PASS', 'issues': None}
    if verdict.startswith("FAIL"):
        return {'status': 'FAIL', 'issues': verdict}
    return {'status': 'FAIL', 'issues': verdict or "FAIL: Unclear verdict from auditor."}

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

    if not isinstance(project_list, list):
        print(f"❌ Error: '{IDEAS_FILE}' must contain a JSON array of projects.")
        return

    # Best-effort: load real platform references (read-only) to reduce guesswork.
    global PLATFORM_ROOT, PLATFORM_REFERENCE
    if PLATFORM_ROOT is None:
        PLATFORM_ROOT = detect_platform_root()
    if PLATFORM_ROOT and not PLATFORM_REFERENCE:
        PLATFORM_REFERENCE = build_platform_reference(PLATFORM_ROOT)

    print(f"📦 Loaded {len(project_list)} projects from {IDEAS_FILE}")

    for idx, inputs in enumerate(project_list):
        if not isinstance(inputs, dict):
            print(f"⚠️ Skipping project {idx+1}: invalid entry (expected object).")
            continue

        project_name = inputs.get("project_name")
        app_idea = inputs.get("app_idea")
        if not isinstance(project_name, str) or not project_name.strip():
            print(f"⚠️ Skipping project {idx+1}: missing/invalid 'project_name'.")
            continue
        if not isinstance(app_idea, str) or not app_idea.strip():
            print(f"⚠️ Skipping project {idx+1} ({project_name}): missing/invalid 'app_idea'.")
            continue

        slug_in = inputs.get("slug")
        app_slug = slugify(slug_in) if isinstance(slug_in, str) and slug_in.strip() else slugify(project_name)

        # CRASH FIX 3: Wrap the entire project in a try/except block
        # If one project explodes, we log it and move to the next.
        try:
            # 1. SETUP PROJECT GLOBALS
            global PROJECT_SETTINGS, state
            
            PROJECT_SETTINGS = {
                "name": project_name.strip(),
                "slug": app_slug,
                "idea": app_idea.strip(),
                "stack": "PHP 8 (vanilla), MySQL (raw PDO via shared/Database.php), HTML5, Tailwind CSS (CDN), Vanilla JS (no build).",
                "design_system": "Modern SaaS. Use Tailwind utility classes. Keep UI consistent with existing apps (simple layouts, clear CTAs).",
                "forbidden_extensions": [".tsx", ".jsx", ".ts", ".py", ".rb", ".go", ".java", ".sh", ".bat"],
                "forbidden_files": ["install.php", "setup.php", "migration.php", "init_db.php", "composer.json", "package.json"]
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
            error_msg = f"🔥 CRASHED ON PROJECT {project_name}: {e}"
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
