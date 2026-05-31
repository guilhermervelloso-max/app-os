# Voice Agent Prototype

This is a simple PC-first voice agent prototype built with:

- a browser HTML UI that captures microphone audio
- a Python FastAPI backend
- the OpenAI Realtime API with `gpt-realtime-2`
- a `web_search` tool path so the assistant can look up current information before answering
- a configurable backend URL so the same frontend can point to a remote server from day one

## What it does

- Streams microphone audio from the browser to Python over WebSocket
- Relays audio to OpenAI Realtime over the official Python SDK
- Plays assistant audio back in the browser
- Lets the model call web search when a question depends on current facts
- Starts listening as soon as you connect, with no wake word required
- Uses the API key from the backend only, so the frontend and Android never need it
- Can point to a remote backend using `VOICE_AGENT_BACKEND_URL`

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set your API key in the backend:

   ```bash
   copy .env.example .env
   ```

   Then edit `.env` and add `OPENAI_API_KEY`.
   If your backend will run remotely, set `VOICE_AGENT_BACKEND_URL` to that public URL.

3. Run the app:

   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

4. Open the frontend in a browser. If `VOICE_AGENT_BACKEND_URL` is set, the page will use that backend automatically.

## Notes

- The prototype uses `server_vad`, so the model can detect when a turn ends.
- Save your API key once in the backend `.env`. After that, the page will try to connect and start listening by itself when it loads.
- The `web_search` tool in `app.py` uses the Responses API to fetch current information, then returns a compact summary to the Realtime session.

## Android next step

The browser protocol stays the same, so the fastest Android adaptation is:

- keep the Python backend unchanged
- point the Android client to the same `VOICE_AGENT_BACKEND_URL`
- send the same JSON messages over WebSocket

That keeps the Realtime integration and the search tool logic reusable across PC and Android.
