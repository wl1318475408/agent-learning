"""工具层：Agent 可调用的外部能力。

每个工具 = 一份 JSON Schema（给模型看的说明书）+ 一个 Python 实现（真正干活）。
模型只负责"决定调用哪个工具、传什么参数"，执行永远由我们的代码完成。
"""
import json
import os
import re
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

import requests


def calculator(expression: str) -> str:
    """安全计算器：白名单字符过滤，杜绝 eval 执行任意代码。"""
    if not re.fullmatch(r"[0-9+\-*/().%\s]+", expression):
        return "错误：只支持数字和 + - * / % () 运算符"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算失败：{e}"


def get_current_time() -> str:
    """返回当前本地时间。模型自己不知道"现在几点"，必须靠工具补全。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


def date_add(days: int, base_date: str = "") -> str:
    """日期推算：在基准日期上加减天数。base_date 留空表示今天。

    这就是第三个测试用例留下的教训：日期运算不是整数运算，
    计算器算不了，必须给模型一个语义匹配的工具。
    """
    try:
        base = datetime.strptime(base_date, "%Y-%m-%d") if base_date else datetime.now()
    except ValueError:
        return "错误：base_date 格式应为 YYYY-MM-DD"
    return (base + timedelta(days=int(days))).strftime("%Y-%m-%d %A")


def web_search(query: str) -> str:
    """联网搜索。有 TAVILY_API_KEY 走 Tavily，否则用 Bing 免 key 兜底。"""
    if os.getenv("TAVILY_API_KEY"):
        return _tavily(query)
    return _bing(query)


def _tavily(query: str) -> str:
    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": 3},
        timeout=15,
    )
    results = resp.json().get("results", [])
    if not results:
        return "未找到相关结果"
    return "\n".join(f"- {r['title']}: {r['content']}（{r['url']}）" for r in results)


def _bing(query: str) -> str:
    resp = requests.get(
        "https://cn.bing.com/search",
        params={"q": query, "format": "rss"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    root = ET.fromstring(resp.text)
    items = root.findall(".//item")[:3]
    if not items:
        return "未找到相关结果"

    lines = []
    for it in items:
        title = it.findtext("title") or ""
        desc = (it.findtext("description") or "").strip()
        link = it.findtext("link") or ""
        lines.append(f"- {title}: {desc[:80]}（{link}）")
    return "\n".join(lines)


# 给模型看的工具说明书：名字和描述写得越清楚，模型调用得越准
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，支持 + - * / % 和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "如 (3+5)*2"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "date_add",
            "description": "日期推算：在某个日期上加减天数并返回结果日期。"
            "凡是“多少天后/前是几号”这类问题必须用本工具，"
            "绝对不要用 calculator 对日期做整数加减。"
            "如果问题里出现了具体日期，必须把它填进 base_date。",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "要增加的天数，负数表示往前推，如 100 或 -7",
                    },
                    "base_date": {
                        "type": "string",
                        "description": "基准日期，格式 YYYY-MM-DD。"
                        "问题中明确给出日期时必须填这个日期；"
                        "只有当问题说的是“今天/现在”时才留空",
                    },
                },
                "required": ["days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "联网搜索实时信息，用于回答训练数据之外或时效性强的问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"],
            },
        },
    },
]

_HANDLERS = {
    "calculator": lambda a: calculator(a["expression"]),
    "get_current_time": lambda a: get_current_time(),
    "date_add": lambda a: date_add(a["days"], a.get("base_date", "")),
    "web_search": lambda a: web_search(a["query"]),
}


def dispatch(name: str, arguments: dict) -> str:
    """按名字路由到具体实现。任何异常都返回字符串，保证循环不中断。"""
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"错误：未知工具 {name}"
    try:
        return handler(arguments)
    except Exception as e:
        return f"工具 {name} 执行失败：{e}"


if __name__ == "__main__":
    # 不经过模型，直接自测工具层
    print(dispatch("calculator", {"expression": "(3+5)*2"}))
    print(dispatch("get_current_time", {}))
    print(dispatch("date_add", {"days": 100}))
    print(dispatch("date_add", {"days": -7, "base_date": "2026-01-01"}))
    print(dispatch("web_search", {"query": "DeepSeek"})[:200])
