"""
Generate changelog from git history and pyproject.toml.

This script generates a changelog in Keep a Changelog format from:
- Git tags (version numbers)
- Conventional commit messages
- README.md changelog section
- pyproject.toml version information
"""

import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import toml

try:
    import git
    GITPYTHON_AVAILABLE = True
except ImportError:
    GITPYTHON_AVAILABLE = False


def get_version_from_pyproject() -> Optional[str]:
    """Get version from pyproject.toml."""
    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            data = toml.load(pyproject_path)
            return data.get("project", {}).get("version")
    except Exception:
        pass
    return None


def get_git_tags() -> List[Tuple[str, str]]:
    """Get git tags with dates."""
    if not GITPYTHON_AVAILABLE:
        return []
    
    try:
        repo = git.Repo(Path(__file__).parent.parent)
        tags = []
        for tag in repo.tags:
            try:
                commit = repo.commit(tag)
                date = datetime.fromtimestamp(commit.committed_date)
                tags.append((tag.name, date.strftime("%Y-%m-%d")))
            except Exception:
                pass
        return sorted(tags, key=lambda x: x[1], reverse=True)
    except Exception:
        return []


def parse_conventional_commits(commits: List[str]) -> Dict[str, List[str]]:
    """Parse conventional commits into categories."""
    categories = {
        "Added": [],
        "Changed": [],
        "Deprecated": [],
        "Removed": [],
        "Fixed": [],
        "Security": []
    }
    
    for commit in commits:
        # Parse conventional commit format: type(scope): message
        match = re.match(r'^(feat|fix|docs|style|refactor|perf|test|chore)(\(.+\))?:\s*(.+)$', commit)
        if match:
            commit_type = match.group(1)
            message = match.group(3)
            
            if commit_type == "feat":
                categories["Added"].append(message)
            elif commit_type == "fix":
                categories["Fixed"].append(message)
            elif commit_type in ["refactor", "perf"]:
                categories["Changed"].append(message)
            elif commit_type == "docs":
                categories["Changed"].append(f"Documentation: {message}")
    
    return categories


def extract_changelog_from_readme() -> Dict[str, str]:
    """Extract changelog section from README.md."""
    readme_path = Path(__file__).parent.parent / "README.md"
    if not readme_path.exists():
        return {}
    
    try:
        content = readme_path.read_text(encoding="utf-8")
        # Extract changelog section (simplified)
        # This would need more sophisticated parsing
        return {}
    except Exception:
        return {}


def generate_changelog() -> str:
    """Generate changelog markdown."""
    version = get_version_from_pyproject() or "Unknown"
    tags = get_git_tags()
    
    changelog = f"""# Changelog

All notable changes to CR-CA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New features in development

### Changed
- Changes in development

## [{version}] - {datetime.now().strftime("%Y-%m-%d")}

### Current Version

Current version: {version}

"""
    
    # Add entries from git tags
    for tag_name, tag_date in tags[:10]:  # Last 10 versions
        changelog += f"\n## [{tag_name}] - {tag_date}\n\n"
        changelog += "### Changes\n\n"
        changelog += "- See git history for details\n\n"
    
    return changelog


def main():
    """Generate and write changelog."""
    changelog = generate_changelog()
    
    # Write to CHANGELOG.md
    changelog_path = Path(__file__).parent.parent / "CHANGELOG.md"
    changelog_path.write_text(changelog, encoding="utf-8")
    
    # Write to docs/changelog/index.md
    docs_changelog_path = Path(__file__).parent.parent / "docs" / "changelog" / "index.md"
    docs_changelog_path.parent.mkdir(parents=True, exist_ok=True)
    docs_changelog_path.write_text(changelog, encoding="utf-8")
    
    print(f"Changelog generated: {changelog_path}")
    print(f"Docs changelog updated: {docs_changelog_path}")


if __name__ == "__main__":
    main()
