const assistantOutput = document.getElementById("assistantOutput");
const backendBaseUrl = window.VOICE_AGENT_CONFIG?.backendUrl || window.location.origin;

let socket = null;
let audioContext = null;
let mediaStream = null;
let mediaSource = null;
let scriptNode = null;
let playingTime = 0;
let isConnecting = false;
let assistantBuffer = "";
let assistantTurnOpen = false;

const TARGET_SAMPLE_RATE = 24000;

function beginAssistantTurn() {
  assistantBuffer = "";
  assistantTurnOpen = true;
}

function appendAssistantText(text) {
  if (!text) return;
  if (!assistantTurnOpen) {
    beginAssistantTurn();
  }
  assistantBuffer += text;
}

function commitAssistantTurn() {
  const finalText = assistantBuffer.trim();
  if (finalText) {
    assistantOutput.textContent = finalText;
  }
  assistantBuffer = "";
  assistantTurnOpen = false;
}

function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
  if (outputSampleRate === inputSampleRate) {
    return buffer;
  }
  if (outputSampleRate > inputSampleRate) {
    throw new Error("Output sample rate must be lower than input sample rate");
  }
  const sampleRateRatio = inputSampleRate / outputSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i += 1) {
      accum += buffer[i];
      count += 1;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult += 1;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function float32ToInt16(float32Array) {
  const out = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i += 1) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function int16ToBase64(int16Array) {
  const bytes = new Uint8Array(int16Array.buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function base64ToInt16(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Int16Array(bytes.buffer);
}

function enqueueAudio(base64) {
  if (!audioContext) return;
  const int16 = base64ToInt16(base64);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i += 1) {
    float32[i] = int16[i] / 0x8000;
  }

  const buffer = audioContext.createBuffer(1, float32.length, TARGET_SAMPLE_RATE);
  buffer.copyToChannel(float32, 0);

  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);

  const now = audioContext.currentTime;
  if (playingTime < now + 0.05) {
    playingTime = now + 0.05;
  }
  source.start(playingTime);
  playingTime += buffer.duration;
}

async function connect() {
  if (isConnecting || (socket && socket.readyState === WebSocket.OPEN)) {
    return;
  }
  isConnecting = true;
  const wsUrl = new URL("/ws", backendBaseUrl);
  wsUrl.protocol = wsUrl.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(wsUrl.toString());

  socket.onopen = async () => {
    audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
    await audioContext.audioWorklet.addModule(
      URL.createObjectURL(
        new Blob(
          [
            `
              class PCMProcessor extends AudioWorkletProcessor {
                process(inputs) {
                  const input = inputs[0] && inputs[0][0];
                  if (input && input.length) {
                    this.port.postMessage(input);
                  }
                  return true;
                }
              }
              registerProcessor("pcm-processor", PCMProcessor);
            `,
          ],
          { type: "application/javascript" }
        )
      )
    );

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    mediaSource = audioContext.createMediaStreamSource(mediaStream);
    scriptNode = new AudioWorkletNode(audioContext, "pcm-processor");
    mediaSource.connect(scriptNode);
    scriptNode.connect(audioContext.destination);

    scriptNode.port.onmessage = (event) => {
      if (socket.readyState !== WebSocket.OPEN) return;
      const input = event.data;
      const downsampled = downsampleBuffer(
        input,
        audioContext.sampleRate,
        TARGET_SAMPLE_RATE
      );
      const pcm16 = float32ToInt16(downsampled);
      socket.send(
        JSON.stringify({
          type: "audio",
          audio: int16ToBase64(pcm16),
        })
      );
    };

    socket.send(JSON.stringify({ type: "start" }));
  };

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === "assistant_text_delta") {
      appendAssistantText(data.delta);
      return;
    }

    if (data.type === "assistant_transcript_delta") {
      return;
    }

    if (data.type === "response_done") {
      commitAssistantTurn();
      return;
    }

    if (data.type === "audio") {
      enqueueAudio(data.delta);
      return;
    }
  };

  socket.onclose = () => {
    cleanupAudio();
    isConnecting = false;
  };

  socket.onerror = () => {
    isConnecting = false;
  };
}

function cleanupAudio() {
  if (scriptNode) {
    scriptNode.disconnect();
    scriptNode = null;
  }
  if (mediaSource) {
    mediaSource.disconnect();
    mediaSource = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (audioContext) {
    void audioContext.close();
    audioContext = null;
  }
  playingTime = 0;
}

connect().catch((error) => {
  assistantOutput.textContent = `Connection error: ${error.message}`;
});
