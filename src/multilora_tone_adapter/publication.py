"""Scan staged, tracked working-tree, and reachable history bytes before publication."""

import re
import subprocess
from pathlib import Path

PATTERNS = {
    'credential': re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{25,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)'),
    'personal_absolute_path': re.compile(r'/(?:Users|home)/[^/\s]+/'),
    'telegram_export': re.compile(r'"(?:from_id|personal_information)"\s*:'),
    'telegram_html_export': re.compile(
        r'''<div\b(?=[^>]*\bclass\s*=\s*["'][^"']*\bmessage\b[^"']*["'])'''
        r'''(?=[^>]*\bid\s*=\s*["']message-?\d+["'])[^>]*>''', re.I,
    ),
}


def _inspect(name: str, content: bytes, source: str) -> list[dict]:
    findings = []
    path = Path(name)
    if ((path.name.startswith('.env') and path.name != '.env.example')
        or path.name in {'auth.json', 'credentials.json', 'secrets.json'}
        or path.name.startswith('config.local.')
        or re.fullmatch(r'messages(?:[0-9]+)?\.html', path.name, re.I)
        or path.name.lower() in {'export_results.html', 'chats.html', 'personal_information.html', 'contacts.html'}
        or path.suffix in {'.jsonl', '.safetensors', '.gguf', '.bin', '.sqlite', '.log'}) or any(
        part in {'exports', 'private', 'runs', 'adapters', 'models', 'data'} for part in path.parts
    ):
        findings.append({'path': name, 'rule': 'private_artifact', 'source': source})
    if len(content) > 1_000_000:
        findings.append({'path': name, 'rule': 'large_file', 'source': source})
    text = content.decode('utf-8', errors='replace')
    for rule, pattern in PATTERNS.items():
        if pattern.search(text):
            if rule in {'telegram_export', 'telegram_html_export'} and name.startswith('tests/'):
                continue  # Reviewed inline synthetic fixtures only.
            findings.append({'path': name, 'rule': rule, 'source': source})
    return findings


def audit(repo: Path) -> dict:
    repo = repo.resolve()
    entries = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=repo).decode().split('\0')
    findings, names, inspected = [], set(), set()
    for entry in filter(None, entries):
        info, name = entry.split('\t', 1)
        mode, blob, stage = info.split()
        names.add(name)
        if mode not in {'100644', '100755'} or stage != '0':
            findings.append({'path': name, 'rule': 'symlink_submodule_or_conflict', 'source': 'index'})
            continue
        staged = subprocess.check_output(['git', 'cat-file', 'blob', blob], cwd=repo)
        findings.extend(_inspect(name, staged, 'index'))
        inspected.add((blob, name))
        path = repo / name
        if path.is_symlink():
            findings.append({'path': name, 'rule': 'symlink', 'source': 'working_tree'})
        elif path.is_file():
            findings.extend(_inspect(name, path.read_bytes(), 'working_tree'))
    history = subprocess.run(['git', 'rev-list', '--objects', '--all'], cwd=repo,
                             capture_output=True, text=True, check=True)
    for line in history.stdout.splitlines():
        if ' ' not in line:
            continue
        blob, name = line.split(' ', 1)
        if (blob, name) in inspected:
            continue
        kind = subprocess.check_output(['git', 'cat-file', '-t', blob], cwd=repo).strip()
        if kind != b'blob':
            continue
        content = subprocess.check_output(['git', 'cat-file', 'blob', blob], cwd=repo)
        findings.extend(_inspect(name, content, 'history'))
        inspected.add((blob, name))
    return {'files': len(names), 'blobs': len(inspected), 'findings': findings,
            'scope': 'index, tracked working-tree files, and all reachable history; review content too'}
