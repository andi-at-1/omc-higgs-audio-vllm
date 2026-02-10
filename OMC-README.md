# OMC Higgs Audio vLLM

Fork von [vLLM](https://github.com/vllm-project/vllm) mit BosonAI Higgs Audio v2 TTS Integration und erweiterten Features.

## Features

### RAS (Repetition Aware Sampling)
Verhindert Audio-Generierungs-Loops durch Blockieren von Tokens die zu häufig wiederholt werden.

- `--ras-window-length N` - Fenstergröße für Wiederholungserkennung (Standard: 7, 0 = deaktiviert)
- `--ras-max-num-repeat N` - Max. erlaubte Wiederholungen bevor Token blockiert wird (Standard: 2)

Kann auch per Request überschrieben werden:
```json
{
  "input": "Text zum Sprechen",
  "ras_window_length": 7,
  "ras_max_num_repeat": 2
}
```

### Seed-Unterstützung
Reproduzierbare Audio-Generierung durch Setzen eines Seeds:
```json
{
  "input": "Text zum Sprechen",
  "seed": 42
}
```

### Speed-Parameter
Anpassung der Sprechgeschwindigkeit (0.25 - 4.0):
```json
{
  "input": "Text zum Sprechen",
  "speed": 1.2
}
```

### Audio-Formate (OpenAI-kompatibel)

Default: `mp3`. Nur PCM wird nativ gestreamt. Alle anderen Formate werden komplett generiert und dann mit korrektem Header encodiert.

| Format | Verhalten | Content-Type |
|--------|-----------|-------------|
| `pcm` | Streaming (headerless) | `audio/pcm` |
| `mp3` | Collect+Encode (Default) | `audio/mpeg` |
| `wav` | Collect+Encode | `audio/wav` |
| `opus` | Collect+Encode | `audio/opus` |
| `aac` | Collect+Encode | `audio/aac` |
| `flac` | Collect+Encode | `audio/flac` |

### Voices Endpoint
Abruf verfügbarer Stimmen:
```bash
curl http://localhost:8778/v1/audio/voices
```

Response:
```json
{
  "voices": [
    {"voice_id": "en_man", "name": "English Man", "language": "en"},
    {"voice_id": "en_woman", "name": "English Woman", "language": "en"},
    {"voice_id": "de_man", "name": "German Man", "language": "de"},
    {"voice_id": "de_woman", "name": "German Woman", "language": "de"},
    {"voice_id": "zh_man", "name": "Chinese Man", "language": "zh"},
    {"voice_id": "zh_woman", "name": "Chinese Woman", "language": "zh"}
  ]
}
```

### Usage Stats
Detaillierte Statistiken in der Response:
```json
{
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 4200,
    "total_tokens": 4215
  },
  "stats": {
    "audio_duration_seconds": 3.5,
    "processing_time_seconds": 1.2,
    "realtime_factor": 2.9
  }
}
```

## Installation

```bash
cd /root/omc-higgs-audio-vllm
python -m venv .venv
source .venv/bin/activate

# Build vLLM from source (mit ccache für schnellere Rebuilds)
pip install -e . --no-build-isolation

# Zusätzliche Dependencies
pip install librosa pydub s3fs boson-multimodal
```

## Server starten

### Mit run.sh (empfohlen)
```bash
./run.sh
```

### Manuell
```bash
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

python -m vllm.entrypoints.bosonai.api_server \
    --model "bosonai/higgs-audio-v2-generation-3B-base" \
    --audio-tokenizer-type "bosonai/higgs-audio-v2-tokenizer" \
    --port 8778 \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --ras-window-length 7 \
    --ras-max-num-repeat 2
```

### Via llama-swap
Konfiguriert in `/root/swap/config.yaml` als `omc-higgs-tts`.

## API Nutzung

### TTS Request
```bash
# MP3 (Default - kein response_format nötig)
curl -X POST "http://localhost:8778/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bosonai/higgs-audio-v2-generation-3B-base",
    "input": "Hallo, dies ist ein Test.",
    "voice": "de_man",
    "seed": 42
  }' \
  --output test.mp3

# WAV
curl -X POST "http://localhost:8778/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bosonai/higgs-audio-v2-generation-3B-base",
    "input": "Hallo, dies ist ein Test.",
    "voice": "de_man",
    "response_format": "wav"
  }' \
  --output test.wav
```

### OpenAI-kompatibel
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8778/v1", api_key="dummy")

response = client.audio.speech.create(
    model="bosonai/higgs-audio-v2-generation-3B-base",
    input="Hallo Welt!",
    voice="de_man",
    response_format="mp3"
)

response.stream_to_file("output.mp3")
```

## GPU-Anforderungen

- ~20 GB VRAM (RTX 3090 empfohlen)
- Mit `--gpu-memory-utilization 0.90` läuft es stabil auf 24GB GPUs

## Geänderte Dateien

| Datei | Änderung |
|-------|----------|
| `vllm/entrypoints/bosonai/api_server.py` | CLI-Args für RAS-Defaults |
| `vllm/entrypoints/bosonai/serving_audio.py` | RAS-Defaults, mm_token_ids Fix, Voices Endpoint |
| `vllm/entrypoints/openai/protocol.py` | RAS, Seed, Speed Parameter |
| `vllm/sampling_params.py` | RAS Parameter |
| `vllm/v1/sample/metadata.py` | RAS Metadata |
| `vllm/v1/sample/sampler.py` | RAS Penalty-Logik mit Bounds-Check |
| `vllm/v1/worker/gpu_input_batch.py` | RAS Batch-Handling |
| `vllm/v1/worker/gpu_model_runner.py` | RAS Integration |
| `run.sh` | Start-Script für GPU 1 |

## Bekannte Fixes

### RAS IndexError bei Audio-Tokens
Problem: `IndexError: index 128016 is out of bounds for dimension 1 with size 1026`

Ursache: RAS Penalty wurde auf Audio-Logits (vocab 1026) mit Text-Token-IDs (vocab ~128k) angewendet.

Fix in `sampler.py`:
```python
vocab_size = logits.shape[1]
for token_id, count in token_counts.items():
    if count >= max_num_repeat and token_id < vocab_size:
        logits[req_index, token_id] = float('-inf')
```

### mm_token_ids Format
Problem: `'list' object has no attribute 'shape'`

Fix in `serving_audio.py`:
```python
if hasattr(output.mm_token_ids, 'shape'):
    audio_tokens = output.mm_token_ids
else:
    audio_tokens = output.mm_token_ids  # Already a list
```
