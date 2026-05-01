from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dashscope import Generation
import json
import os
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)

DASHSCOPE_API_KEY =os.getenv("DASHSCOPE—API—KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请在Codespaces Settings中设置DASHSCOPE_API_KEY Secret")

SYSTEM_PROMPT = """
我叫小杨，是一个考研助手，可以回答各种考研问题以及学习规划，语气亲切耐心，只回答考研相关问题。
"""

# 保存评测日志
LOG_FILE = "llm_evaluate_log.txt"


def write_log(log_info):
    """写入评测日志"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_info, ensure_ascii=False) + "\n")


@app.route("/api/chat", methods=["POST"])
def chat():
    start_time = time.time()  # 整个请求开始时间
    data = request.get_json()
    user_messages = data.get("messages", [])
    model = data.get("model", "qwen-turbo")

    if not user_messages:
        return jsonify({"error": "消息不能为空"}), 400

    # 拼接人设
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages

    # 评测变量
    first_token_time = None
    full_content = ""

    def generate():
        nonlocal first_token_time, full_content
        responses = Generation.call(
            model=model,
            messages=messages,
            result_format="message",
            stream=True,
            incremental_output=True
        )
        for resp in responses:
            # 记录首Token耗时
            if first_token_time is None:
                first_token_time = time.time()
            if resp.output.choices:
                content = resp.output.choices[0].message.content
                full_content = content
                yield f"data: {json.dumps({'content': content})}\n\n"

        # 全部生成完成，统计指标
        end_time = time.time()
        total_cost = round(end_time - start_time, 3)
        first_delay = round(first_token_time - start_time, 3) if first_token_time else 0

        # 构造日志
        log_info = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "user_query": user_messages[-1]["content"],
            "llm_reply": full_content,
            "first_token_delay_s": first_delay,  # 首token时延
            "total_cost_s": total_cost,  # 总耗时
        }
        write_log(log_info)

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)