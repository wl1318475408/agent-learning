"""会话记忆：把对话历史写到本地文件，重启后接着聊。

只做一件事——history 列表的存取，不掺任何业务逻辑。
"""
import json
from pathlib import Path

SESSION_DIR = Path(__file__).parent / "sessions"


def _path(session_id: str) -> Path:
    SESSION_DIR.mkdir(exist_ok=True)
    return SESSION_DIR / f"{session_id}.json"


def load_history(session_id: str) -> list:
    """读不到就返回空列表，让程序照常启动。"""
    path = _path(session_id)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_history(session_id: str, history: list) -> None:
    """每轮结束落盘一次。"""
    _path(session_id).write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
