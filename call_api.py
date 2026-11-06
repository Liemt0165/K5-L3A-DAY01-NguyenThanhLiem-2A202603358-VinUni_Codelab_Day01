"""
Day 1 — LLM API Foundation: All-in-One API Caller (OpenAI, Gemini, Anthropic, Streaming)
AICB-P1: AI Practical Competency Program, Phase 1

File này gộp toàn bộ triển khai từ 4 file task:
  - Task 1: task1_call_openai.py
  - Task 2: task2_call_gemini.py
  - Task 3: task3_call_anthropic.py
  - Task 4: task4_streaming.py
Định dạng theo mẫu của starter-code/template.py
"""

import os
import time
from typing import Any, Callable
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
import anthropic

# Tải cấu hình từ file .env
load_dotenv()

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-flash-lite": {"input": 0.0375, "output": 0.150},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers (giữ theo model đang dùng trong các file task)
OPENAI_MODEL = "gpt-5.4"
GEMINI_MODEL = "gemini-2.5-flash-lite"
ANTHROPIC_MODEL = "claude-opus-5"
STREAMING_MODEL = "gpt-5.4"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-5.4 / OpenAI-Compatible Gateway)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.4,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-5.4).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')
    """
    # Khởi tạo client kết nối tới Base URL và API Key từ file .env
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # Bắt đầu đo thời gian phản hồi (latency)
    start_time = time.time()

    # Gọi Chat Completions API
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    latency = time.time() - start_time

    # Trích xuất nội dung phản hồi
    response_text = response.choices[0].message.content or ""

    # Trích xuất số lượng token đã sử dụng
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }

    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini (Gemini 2.5 Flash Lite)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.4,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash Lite as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash-lite).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')
    """
    # Khởi tạo client kết nối tới Base URL và API Key từ file .env
    base_url = os.getenv("GEMINI_BASE_URL")
    http_options = types.HttpOptions(base_url=base_url) if base_url else None

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=http_options,
    )

    # Cấu hình tham số sinh văn bản
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    # Bắt đầu đo thời gian phản hồi (latency)
    start_time = time.time()

    # Gọi API sinh nội dung
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    latency = time.time() - start_time

    # Trích xuất nội dung phản hồi
    response_text = response.text or ""

    # Trích xuất số lượng token đã sử dụng
    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
        "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
    }

    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Claude Opus 5)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.4,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude Opus 5 as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-opus-5).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')
    """
    # Khởi tạo client kết nối tới Base URL và API Key từ file .env
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    # Nếu base_url kết thúc bằng '/v1', bỏ bớt vì SDK Anthropic tự động thêm '/v1/messages'
    if base_url and base_url.endswith("/v1"):
        base_url = base_url[:-3]

    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url=base_url,
    )

    # Chuẩn bị tham số cho messages.create (hỗ trợ SDK anthropic 2.x+)
    extra_body = {}
    if temperature is not None:
        extra_body["temperature"] = temperature
    if top_p is not None:
        extra_body["top_p"] = top_p

    # Bắt đầu đo thời gian phản hồi (latency)
    start_time = time.time()

    # Gọi API tạo tin nhắn
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        extra_body=extra_body if extra_body else None,
    )

    latency = time.time() - start_time

    # Trích xuất nội dung phản hồi
    response_text = response.content[0].text if response.content else ""

    # Trích xuất số lượng token đã sử dụng
    usage = {
        "input_tokens": response.usage.input_tokens if response.usage else 0,
        "output_tokens": response.usage.output_tokens if response.usage else 0,
    }

    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Streaming Token & Chatbot (OpenAI-Compatible Gateway)
# ---------------------------------------------------------------------------
def stream_openai(
    prompt: str,
    model: str = STREAMING_MODEL,
    temperature: float = 0.4,
) -> tuple[str, float, dict, float]:
    """
    Stream responses from OpenAI Chat Completions API chunk by chunk,
    tính số lượng tokens và tốc độ sinh (tokens/giây).

    Args:
        prompt:      The user message to send.
        model:       The model to use (default: gpt-5.4).
        temperature: Sampling temperature.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')
            - tokens_per_second (float)
    """
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    start_time = time.time()

    # Thử yêu cầu trả về thông tin usage trong stream (nếu proxy/server hỗ trợ)
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
    except Exception:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=True,
        )

    full_response = ""
    chunk_count = 0
    usage_data = None

    # Nhận và in từng token (chunk) ra màn hình theo thời gian thực
    for chunk in stream:
        if hasattr(chunk, "usage") and chunk.usage:
            usage_data = {
                "input_tokens": chunk.usage.prompt_tokens or 0,
                "output_tokens": chunk.usage.completion_tokens or 0,
            }

        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            print(delta, end="", flush=True)
            full_response += delta
            chunk_count += 1

    latency = time.time() - start_time
    print()  # Xuống dòng sau khi stream kết thúc

    # Nếu proxy không gửi kèm usage qua stream, ước lượng số token
    if not usage_data or usage_data["output_tokens"] == 0:
        estimated_output = max(chunk_count, round(len(full_response.split()) * 1.3))
        estimated_input = max(1, round(len(prompt.split()) * 1.3))
        usage_data = {
            "input_tokens": estimated_input,
            "output_tokens": estimated_output,
        }

    output_tokens = usage_data["output_tokens"]
    tokens_per_second = (output_tokens / latency) if latency > 0 else 0.0

    return full_response, latency, usage_data, tokens_per_second


def streaming_chatbot(
    model: str = STREAMING_MODEL,
    temperature: float = 0.4,
    max_history_turns: int = 3,
) -> None:
    """
    Run an interactive streaming chatbot in the terminal using OpenAI streaming.

    Behaviour:
        - Streams response tokens as they arrive.
        - Maintains the last 3 turns of conversation history for context (max 6 messages).
        - Typing 'quit' hoặc 'exit' kết thúc session.
    """
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    history: list[dict[str, str]] = []

    print("=" * 60)
    print(f"🤖 STREAMING CHATBOT ĐÃ SẴN SÀNG (Model: {model})")
    print("Nhập 'quit' hoặc 'exit' để kết thúc phiên trò chuyện.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n👤 Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nĐã thoát phiên trò chuyện.")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            print("Đã kết thúc phiên trò chuyện. Tạm biệt!")
            break

        history.append({"role": "user", "content": user_input})
        max_messages = max_history_turns * 2
        active_messages = history[-max_messages:]

        print(f"\n🤖 {model}: ", end="", flush=True)

        start_time = time.time()
        assistant_reply = ""
        chunk_count = 0
        usage_data = None

        try:
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=active_messages,
                    temperature=temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            except Exception:
                stream = client.chat.completions.create(
                    model=model,
                    messages=active_messages,
                    temperature=temperature,
                    stream=True,
                )

            for chunk in stream:
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = {
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                    }
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    print(delta, end="", flush=True)
                    assistant_reply += delta
                    chunk_count += 1

            latency = time.time() - start_time
            print()

            if not usage_data or usage_data["output_tokens"] == 0:
                usage_data = {"output_tokens": max(chunk_count, round(len(assistant_reply.split()) * 1.3))}

            out_toks = usage_data["output_tokens"]
            speed = (out_toks / latency) if latency > 0 else 0.0
            print(f"   [⏱️ {latency:.2f}s | 📊 {out_toks} tokens | ⚡ {speed:.2f} tok/s]")

            history.append({"role": "assistant", "content": assistant_reply})

        except Exception as e:
            print(f"\n❌ Lỗi khi stream: {e}")


# ---------------------------------------------------------------------------
# Entry point: Chạy kiểm thử cả 4 Task với prompt dùng chung
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Prompt dùng chung cho toàn bộ các task
    input_prompt = "giải thích chi tiết LLM"

    print("=" * 70)
    print("🚀 BẮT ĐẦU CHẠY KIỂM THỬ TẤT CẢ 4 TASK (call_api.py)")
    print(f"📝 Prompt dùng chung: '{input_prompt}'")
    print("=" * 70)

    # 1. TASK 1: OpenAI
    print("\n" + "=" * 70)
    print(f"🔹 [TASK 1] CALL OPENAI - Model: [{OPENAI_MODEL}]")
    print("-" * 70)
    try:
        reply_oa, lat_oa, use_oa = call_openai(prompt=input_prompt, model=OPENAI_MODEL)
        print(reply_oa)
        print("-" * 70)
        print(f"⏱️ Latency: {lat_oa:.3f}s | 📊 In: {use_oa['input_tokens']} | Out: {use_oa['output_tokens']} | Total: {use_oa['input_tokens'] + use_oa['output_tokens']}")
    except Exception as e:
        print(f"❌ Lỗi Task 1: {e}")

    # 2. TASK 2: Google Gemini
    print("\n" + "=" * 70)
    print(f"🔹 [TASK 2] CALL GEMINI - Model: [{GEMINI_MODEL}]")
    print("-" * 70)
    try:
        reply_gm, lat_gm, use_gm = call_gemini(prompt=input_prompt, model=GEMINI_MODEL)
        print(reply_gm)
        print("-" * 70)
        print(f"⏱️ Latency: {lat_gm:.3f}s | 📊 In: {use_gm['input_tokens']} | Out: {use_gm['output_tokens']} | Total: {use_gm['input_tokens'] + use_gm['output_tokens']}")
    except Exception as e:
        print(f"❌ Lỗi Task 2: {e}")

    # 3. TASK 3: Anthropic Claude
    print("\n" + "=" * 70)
    print(f"🔹 [TASK 3] CALL ANTHROPIC - Model: [{ANTHROPIC_MODEL}]")
    print("-" * 70)
    try:
        reply_an, lat_an, use_an = call_anthropic(prompt=input_prompt, model=ANTHROPIC_MODEL)
        print(reply_an)
        print("-" * 70)
        print(f"⏱️ Latency: {lat_an:.3f}s | 📊 In: {use_an['input_tokens']} | Out: {use_an['output_tokens']} | Total: {use_an['input_tokens'] + use_an['output_tokens']}")
    except Exception as e:
        print(f"❌ Lỗi Task 3: {e}")

    # 4. TASK 4: Streaming OpenAI
    print("\n" + "=" * 70)
    print(f"🔹 [TASK 4] STREAMING OPENAI - Model: [{STREAMING_MODEL}]")
    print("-" * 70)
    try:
        reply_st, lat_st, use_st, speed_st = stream_openai(prompt=input_prompt, model=STREAMING_MODEL)
        print("-" * 70)
        print(f"⏱️ Latency: {lat_st:.3f}s | 📊 In: {use_st['input_tokens']} | Out: {use_st['output_tokens']} | Total: {use_st['input_tokens'] + use_st['output_tokens']} | ⚡ Speed: {speed_st:.2f} tok/s")
    except Exception as e:
        print(f"❌ Lỗi Task 4: {e}")

