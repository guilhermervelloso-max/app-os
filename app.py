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
WEB_SEARCH_MODEL = os.getenv("OPENAI_WEB_SEARCH_MODEL", "gpt-5.5")
VOICE_AGENT_BACKEND_URL = os.getenv("VOICE_AGENT_BACKEND_URL", "")

SYSTEM_PROMPT = """You are a warm, concise voice assistant.

Speak naturally and keep replies short unless the user asks for detail.
Use the web_search tool whenever the answer depends on current facts, recent events,
prices, schedules, availability, releases, or any information that may have changed.

If you use web search, base your answer on the latest information available and say
that you checked the web. If information is uncertain, say so plainly.

When the user asks something that depends on the present moment, prefer exact dates
and concrete facts over vague relative language.
"""

WEB_SEARCH_TOOL = {
    "type": "function",
    "name": "web_search",
    "description": "Search the internet for up-to-date information and return a concise, sourced summary.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web.",
            },
            "max_results": {
                "type": "integer",
                "description": "How many relevant results to keep in the summary.",
                "minimum": 1,
                "maximum": 5,
                "default": 3,
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
    """Use the Responses API web_search tool to get a current answer."""
    prompt = (
        "Search the web for the user's question and summarize the most useful current "
        "information in a compact, factual way. Include source URLs if available.\n\n"
        f"Question: {query}"
    )

    client = make_client()
    response = await client.responses.create(
        model=WEB_SEARCH_MODEL,
        input=prompt,
        tools=[{"type": "web_search"}],
    )

    text = getattr(response, "output_text", "") or ""
    text = text.strip()

    if not text:
        text = "I searched the web, but I could not extract a readable summary."

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > max_results + 2:
        lines = lines[: max_results + 2]
        lines.append("...")  # keep the tool result compact for the voice model

    return {
        "query": query,
        "summary": "\n".join(lines),
        "model": WEB_SEARCH_MODEL,
    }


async def send_text(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_text(_safe_json(payload))


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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
                    "type": "realtime",
                    "model": MODEL,
                    "voice": VOICE,
                    "instructions": SYSTEM_PROMPT,
                    "output_modalities": ["audio", "text"],
                    "audio": {
                        "input": {
                            "turn_detection": {
                                "type": "server_vad",
                            }
                        }
                    },
                    "tools": [WEB_SEARCH_TOOL],
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

                            if tool_name == "web_search":
                                result = await web_search(
                                    query=str(arguments.get("query", "")),
                                    max_results=int(arguments.get("max_results", 3)),
                                )
                            else:
                                result = {
                                    "error": f"Unsupported tool: {tool_name}",
                                }

                            await connection.conversation.item.create(
                                item={
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": _safe_json(result),
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
