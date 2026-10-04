"""Skills: instruction files a persona can load by name.

A skill is instructions, not code. Split out of personas.py so the persona
model stays a pure data type and the filesystem lookup has one home.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger("service-desk.personas")

PROJECT_ROOT = Path(__file__).parents[2]
SKILLS_DIR = Path(os.getenv("SKILLS_DIR") or PROJECT_ROOT / "skills")


def load_skill(name: str) -> str:
    """A skill is instructions, not code. Missing ones are skipped loudly rather
    than failing the call: an agent with one fewer skill still answers.
    Supports both `skills/<name>.md` and standard Agent Skills directory format `skills/<name>/SKILL.md`.
    """
    # 1. Check directory format: skills/<name>/SKILL.md
    dir_path = SKILLS_DIR / name / "SKILL.md"
    if dir_path.is_file():
        try:
            return dir_path.read_text(encoding="utf-8").strip()
        except Exception as ex:
            logger.warning("Error reading skill at %s: %s", dir_path, ex)

    # 2. Check flat file format: skills/<name>.md
    path = SKILLS_DIR / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning("skill %r not found at %s or %s, continuing without it", name, path, dir_path)
        return ""


def get_skill_manifest(name: str) -> dict[str, str]:
    """Extracts summary metadata from a skill for compact progressive indexing."""
    raw = load_skill(name)
    if not raw:
        return {"name": name, "description": ""}
    
    # Check for frontmatter
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            desc = ""
            for line in frontmatter.splitlines():
                if line.strip().startswith("description:"):
                    desc = line.split(":", 1)[1].strip()
                    break
            return {"name": name, "description": desc, "body": parts[2].strip()}
            
    # Fallback to first non-empty line
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    first_line = lines[0] if lines else ""
    return {"name": name, "description": first_line.lstrip("#").strip(), "body": raw}
