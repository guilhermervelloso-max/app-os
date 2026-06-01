from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE, override=False)

MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2")
VOICE = os.getenv("OPENAI_REALTIME_VOICE", "alloy")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
VOICE_AGENT_BACKEND_URL = os.getenv("VOICE_AGENT_BACKEND_URL", "")

SYSTEM_PROMPT = """You are a warm, concise voice assistant with access to real-time web search.

You have a tool called lookup_current_info. Use it for any question about:
- prices (crypto, stocks, currency, goods)
- news or current events
- sports scores or standings
- weather
- any information that changes over time

Call the tool, then give a short spoken answer based on the result you received.
For timeless topics (math, history, how-to) answer directly.
Keep replies concise and natural.
"""

WEB_SEARCH_TOOL = {
    "type": "function",
    "name": "lookup_current_info",
    "description": "Retrieves up-to-date information on any topic by querying a live data source.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The topic or question to look up.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")

app = FastAPI(title="Voice Agent Prototype")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_runtime_env_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value else default


def get_api_key() -> str | None:
    value = get_runtime_env_value("OPENAI_API_KEY")
    return value if value else None


def make_client() -> AsyncOpenAI:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured yet.")
    return AsyncOpenAI(api_key=api_key)


def get_config_summary() -> dict[str, Any]:
    return {
        "ok": True,
        "model": get_runtime_env_value("OPENAI_REALTIME_MODEL", "gpt-realtime-2"),
        "voice": get_runtime_env_value("OPENAI_REALTIME_VOICE", "alloy"),
        "web_search_model": get_runtime_env_value("OPENAI_WEB_SEARCH_MODEL", "gpt-5.5"),
        "api_key_configured": bool(get_api_key()),
        "backend_url": VOICE_AGENT_BACKEND_URL,
    }


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _decode_audio_chunk(data_b64: str) -> bytes:
    return base64.b64decode(data_b64)


def _encode_audio_chunk(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


async def web_search(query: str, max_results: int = 3) -> dict[str, Any]:
    if not PERPLEXITY_API_KEY:
        return {"error": "PERPLEXITY_API_KEY not configured", "query": query}
    client = AsyncOpenAI(
        api_key=PERPLEXITY_API_KEY,
        base_url="https://api.perplexity.ai",
    )
    response = await client.chat.completions.create(
        model="sonar",
        messages=[
            {
                "role": "system",
                "content": "Search the internet for the most recent, up-to-date information. Always use real-time web search results, not your training knowledge.",
            },
            {"role": "user", "content": query},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        text = "No results found."
    return {"query": query, "summary": text, "model": "sonar"}


async def send_text(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_text(_safe_json(payload))


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/test-search")
async def test_search(q: str = "preço do bitcoin hoje") -> JSONResponse:
    try:
        result = await web_search(q)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc), "type": type(exc).__name__}, status_code=500)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(get_config_summary())


@app.get("/api/config")
async def api_config() -> JSONResponse:
    return JSONResponse(get_config_summary())


@app.get("/config.js")
async def config_js() -> Response:
    script = (
        "window.VOICE_AGENT_CONFIG = "
        + _safe_json({"backendUrl": VOICE_AGENT_BACKEND_URL})
        + ";\n"
    )
    return Response(content=script, media_type="application/javascript")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    await send_text(
        websocket,
        {
            "type": "status",
            "state": "connecting",
            "message": "Connecting to OpenAI Realtime...",
        },
    )

    try:
        client = make_client()
        async with client.realtime.connect(model=MODEL) as connection:
            pending_tool_calls: dict[str, dict[str, Any]] = {}

            await connection.session.update(
                session={
                    "modalities": ["audio", "text"],
                    "voice": VOICE,
                    "instructions": SYSTEM_PROMPT,
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.75,
                    },
                    "tools": [WEB_SEARCH_TOOL],
                    "tool_choice": "auto",
                }
            )

            await send_text(
                websocket,
                {
                    "type": "status",
                    "state": "ready",
                    "message": f"Connected to {MODEL}.",
                },
            )

            async def browser_to_openai() -> None:
                while True:
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "audio":
                        chunk = _decode_audio_chunk(data["audio"])
                        await connection.input_audio_buffer.append(
                            audio=_encode_audio_chunk(chunk)
                        )
                        continue

                    if msg_type == "start":
                        await send_text(
                            websocket,
                            {
                                "type": "status",
                                "state": "recording",
                                "message": "Microphone streaming started.",
                            },
                        )
                        continue

                    if msg_type == "stop":
                        await connection.input_audio_buffer.commit()
                        await connection.response.create()
                        await send_text(
                            websocket,
                            {
                                "type": "status",
                                "state": "thinking",
                                "message": "Turn committed. Waiting for the assistant.",
                            },
                        )
                        continue

                    if msg_type == "clear":
                        await connection.input_audio_buffer.clear()
                        await send_text(
                            websocket,
                            {
                                "type": "status",
                                "state": "idle",
                                "message": "Audio buffer cleared.",
                            },
                        )
                        continue

            async def openai_to_browser() -> None:
                response_audio_started = False
                async for event in connection:
                    event_type = getattr(event, "type", "")

                    if event_type == "session.created":
                        await send_text(
                            websocket,
                            {
                                "type": "session",
                                "state": "created",
                                "session_id": getattr(event.session, "id", None),
                            },
                        )
                        continue

                    if event_type == "session.updated":
                        await send_text(
                            websocket,
                            {
                                "type": "session",
                                "state": "updated",
                                "session_id": getattr(event.session, "id", None),
                            },
                        )
                        continue

                    if event_type == "response.output_audio.delta":
                        if not response_audio_started:
                            response_audio_started = True
                            await connection.input_audio_buffer.clear()
                        await send_text(
                            websocket,
                            {
                                "type": "audio",
                                "delta": event.delta,
                                "item_id": getattr(event, "item_id", None),
                            },
                        )
                        continue

                    if event_type == "response.output_audio_transcript.delta":
                        await send_text(
                            websocket,
                            {
                                "type": "assistant_transcript_delta",
                                "delta": event.delta,
                                "item_id": getattr(event, "item_id", None),
                            },
                        )
                        continue

                    if event_type == "response.output_text.delta":
                        await send_text(
                            websocket,
                            {
                                "type": "assistant_text_delta",
                                "delta": event.delta,
                                "item_id": getattr(event, "item_id", None),
                            },
                        )
                        continue

                    if event_type == "response.output_item.done":
                        item = getattr(event, "item", None)
                        item_type = getattr(item, "type", None)
                        if item_type == "function_call":
                            item_id = getattr(item, "id", None)
                            call_id = (
                                getattr(item, "call_id", None)
                                or item_id
                            )
                            pending = pending_tool_calls.get(item_id or "", {})
                            tool_name = getattr(item, "name", "") or pending.get("name", "")
                            raw_arguments = getattr(item, "arguments", "") or pending.get("arguments", "")
                            if not raw_arguments:
                                raw_arguments = "{}"
                            try:
                                arguments = json.loads(raw_arguments)
                            except json.JSONDecodeError:
                                arguments = {"raw": raw_arguments}

                            await send_text(
                                websocket,
                                {
                                    "type": "tool_call",
                                    "name": tool_name,
                                    "call_id": call_id,
                                    "arguments": arguments,
                                },
                            )

                            if tool_name == "lookup_current_info":
                                try:
                                    result = await web_search(
                                        query=str(arguments.get("query", "")),
                                    )
                                except Exception as exc:
                                    logger.exception("lookup_current_info failed for query %r", arguments.get("query", ""))
                                    result = {"error": str(exc), "query": arguments.get("query", "")}
                            else:
                                result = {
                                    "error": f"Unsupported tool: {tool_name}",
                                }

                            await connection.conversation.item.create(
                                item={
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": result.get("summary") or result.get("error", "No results found."),
                                }
                            )
                            await connection.response.create()

                            await send_text(
                                websocket,
                                {
                                    "type": "tool_result",
                                    "name": tool_name,
                                    "call_id": call_id,
                                    "result": result,
                                },
                            )
                        continue

                    if event_type == "response.function_call_arguments.done":
                        pending_tool_calls[getattr(event, "item_id", "")] = {
                            "name": getattr(event, "name", ""),
                            "arguments": getattr(event, "arguments", "{}") or "{}",
                        }
                        continue

                    if event_type == "error":
                        await send_text(
                            websocket,
                            {
                                "type": "error",
                                "message": getattr(event.error, "message", "Unknown error"),
                                "code": getattr(event.error, "code", None),
                                "error_type": getattr(event.error, "type", None),
                            },
                        )
                        continue

                    if event_type == "response.done":
                        response_audio_started = False
                        await send_text(
                            websocket,
                            {
                                "type": "response_done",
                                "response_id": getattr(event, "response", None)
                                and getattr(event.response, "id", None),
                            },
                        )
                        continue

            browser_task = asyncio.create_task(browser_to_openai())
            openai_task = asyncio.create_task(openai_to_browser())

            done, pending = await asyncio.wait(
                {browser_task, openai_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )

            for task in pending:
                task.cancel()
            for task in done:
                with suppress(asyncio.CancelledError):
                    task.result()

    except WebSocketDisconnect:
        logger.info("Browser disconnected")
    except Exception as exc:  # pragma: no cover - defensive logging for runtime use
        logger.exception("Voice bridge failed: %s", exc)
        with suppress(Exception):
            await send_text(
                websocket,
                {
                    "type": "error",
                    "message": str(exc),
                },
            )
    finally:
        with suppress(Exception):
            await websocket.close()
