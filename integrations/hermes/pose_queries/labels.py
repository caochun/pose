"""User-assigned names for appearance profile IDs."""
from __future__ import annotations

import re
import sqlite3
import time
import unicodedata
from pathlib import Path

DB=Path('/root/Develop/pose/person_profiles.sqlite3')


def names():
    if not DB.exists():
        return {}
    with sqlite3.connect(f'file:{DB}?mode=ro',uri=True,timeout=2) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='profile_names'").fetchone():
            return {}
        return dict(db.execute('SELECT profile_id,name FROM profile_names'))


def display(pid, mapping=None):
    mapping=names() if mapping is None else mapping
    name=mapping.get(int(pid))
    return f'{name}（P{pid}）' if name else f'P{pid}'


def resolve(arg):
    arg=arg.strip()
    if re.fullmatch(r'[Pp]?\d+',arg):
        return int(arg.lstrip('Pp'))
    key=unicodedata.normalize('NFKC',arg).casefold()
    for pid,name in names().items():
        if unicodedata.normalize('NFKC',name).casefold()==key:
            return pid
    raise ValueError('没有找到该名称，请使用 /人物 P3 或先用 /命名 P3 名称。')


def rename(args=''):
    parts=args.strip().split(maxsplit=1)
    mapping=names()
    if not parts:
        lines=['用法：/命名 P3 阿春；改名再次执行；/命名 P3 - 清除名称。']
        lines.extend(display(pid,mapping) for pid in sorted(mapping)[:30])
        return '\n'.join(lines)
    if not re.fullmatch(r'[Pp]?\d+',parts[0]):
        return '用法：/命名 P3 阿春；/命名 P3 - 清除名称。'
    pid=int(parts[0].lstrip('Pp'))
    with sqlite3.connect(f'file:{DB}?mode=rw',uri=True,timeout=3) as db:
        if not db.execute('SELECT 1 FROM profiles WHERE id=?',(pid,)).fetchone():
            return f'P{pid}档案不存在，请先用 /人物 查看编号。'
        if len(parts)==1:
            return display(pid,mapping)+('，尚未命名。' if pid not in mapping else '，可用 /命名 修改。')
        name=unicodedata.normalize('NFKC',parts[1]).strip()
        if name!='-' and (not re.fullmatch(r'[\w .·-]{1,24}',name) or re.fullmatch(r'[Pp]?\d+',name)):
            return '名称限1–24字，可用中文、字母、数字、空格、点、下划线和短横线；不能只用P编号或数字。'
        db.execute('''CREATE TABLE IF NOT EXISTS profile_names (
            profile_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            name_key TEXT NOT NULL UNIQUE, updated_at REAL NOT NULL)''')
        if name=='-':
            db.execute('DELETE FROM profile_names WHERE profile_id=?',(pid,))
            return f'已清除P{pid}的名称，人物记录仍保留。'
        try:
            db.execute('INSERT INTO profile_names(profile_id,name,name_key,updated_at) VALUES(?,?,?,?) '
                       'ON CONFLICT(profile_id) DO UPDATE SET name=excluded.name,name_key=excluded.name_key,updated_at=excluded.updated_at',
                       (pid,name,name.casefold(),time.time()))
        except sqlite3.IntegrityError:
            return '这个名称已绑定其他P编号，请使用不同名称。'
    return f'已命名：{name}（P{pid}）。可用 /人物 {name} 查询。名称绑定此档案，外观匹配仍可能出错。'


def decorate_events(text):
    mapping=names()
    return re.sub(r'\bP(\d+)\b',lambda match:display(int(match[1]),mapping),text)
