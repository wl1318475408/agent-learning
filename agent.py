"""最小 Agent：LLM + 工具 + 循环决策（ReAct 思想，基于原生 tool calling）。

核心循环只有三步：
    1. 把对话历史发给模型
    2. 模型要么直接回复（任务完成），要么要求调用工具
    3. 我们执行工具、把结果追加进历史，回到第 1 步
所谓 Agent 的"智能"就藏在这个循环里 —— 框架做的也只是把它包装得更花哨。
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from memory import load_history, save_history
from tools import TOOLS, dispatch

load_dotenv()  # 从 .env 读取配置

MAX_STEPS = 10  # 保险丝：防止模型陷入无限工具循环烧钱，工程上必须有

SYSTEM_PROMPT = (
    "你是一个能干的中文助手。"
    "涉及计算、日期推算、当前时间或实时信息时，必须调用工具，不要凭记忆编造。"
    "日期加减一律用 date_add，不要用 calculator 把日期当整数算。"
)

_client = None


def get_client() -> OpenAI:
    """懒加载客户端：缺 Key 时给出可操作的报错，而不是一堆堆栈。"""
    global _client
    if _client is None:
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise SystemExit("缺少 LLM_API_KEY：请复制 .env.example 为 .env 并填入你的 Key")
        _client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        )
    return _client


def run_agent(user_input: str, history: list) -> str:
    """执行一轮完整任务并返回最终答案。history 跨轮保留，即短期记忆。"""
    history.append({"role": "user", "content": user_input})
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    for step in range(1, MAX_STEPS + 1):
        response = get_client().chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
            tools=TOOLS,
        )
        message = response.choices[0].message

        # 情况 1：模型没调工具 → 这就是最终答案，循环结束
        if not message.tool_calls:
            history.append({"role": "assistant", "content": message.content})
            return message.content

        # 情况 2：模型要求调工具 → 逐个执行，结果喂回历史，继续循环
        history.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            print(f"  [step {step}] 调用工具 {call.function.name}({args})")
            result = dispatch(call.function.name, args)
            print(f"           结果：{result[:120]}")
            history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    return f"已达到最大步数 {MAX_STEPS}，任务中止（保险丝触发）"


def main() -> None:
    session_id = "default"  # 会话名，未来可以按用户/日期分文件
    history = load_history(session_id)

    print("最小 Agent 已启动，输入 exit 退出。")
    print(f"模型：{os.getenv('LLM_MODEL', 'deepseek-chat')}，工具：3 个")
    print(f"已恢复 {len(history)} 条历史消息\n")

    while True:
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        answer = run_agent(user_input, history)
        save_history(session_id, history)  # 落盘，下一轮重启还能读到
        print(f"\nAgent > {answer}\n")


if __name__ == "__main__":
    main()
