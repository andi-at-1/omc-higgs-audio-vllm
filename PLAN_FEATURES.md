# Plan: Feature-Erweiterungen für Higgs Audio vLLM

## Implementierungsstatus: ✅ ABGESCHLOSSEN

Alle Features wurden implementiert und sind syntaktisch verifiziert.

## Übersicht

Implementierung fehlender Features für die Audio Speech API:

1. ✅ **RAS (Repetition Aware Sampling)** - Verhindert Audio-Loops
2. ✅ **Seed-Handling** - Reproduzierbare Generierung
3. ✅ **`/v1/audio/voices` Endpoint** - API für verfügbare Stimmen
4. ✅ **Usage Stats** - Token-Statistiken (geloggt + im Request-State)
5. ✅ **Speed Parameter** - Sprechgeschwindigkeit anpassen (0.25-4.0x)

### Neue Parameter
| Parameter | Beschreibung | Typischer Wert |
|-----------|--------------|----------------|
| `ras_window_length` | Fenstergröße für Wiederholungserkennung | 7 |
| `ras_max_num_repeat` | Max. erlaubte Wiederholungen im Fenster | 2 |
| `seed` | Random Seed für reproduzierbare Generierung | int |
| `speed` | Sprechgeschwindigkeit (0.5 = langsam, 2.0 = schnell) | 1.0 |

---

## Teil 1: RAS Parameter

### 1.1 Request-Ebene: `AudioSpeechRequest`

**Datei:** `vllm/entrypoints/openai/protocol.py` (Zeile ~1743)

**Änderungen:**
```python
class AudioSpeechRequest(OpenAIBaseModel):
    # ... bestehende Felder ...

    # NEU: RAS Parameter
    ras_window_length: Optional[int] = None
    """ Window size for repetition detection (e.g., 7) """

    ras_max_num_repeat: Optional[int] = None
    """ Max allowed repetitions within window (e.g., 2) """

    seed: Optional[int] = None
    """ Random seed for reproducible generation """
```

**Methode `to_sampling_params()` anpassen:**
- RAS-Parameter und Seed an `SamplingParams` durchreichen

---

### 1.2 Core: `SamplingParams`

**Datei:** `vllm/sampling_params.py` (Zeile ~108)

**Änderungen:**
```python
class SamplingParams(msgspec.Struct, ...):
    # ... bestehende Felder ...

    # NEU: RAS Parameter
    ras_window_length: Optional[int] = None
    ras_max_num_repeat: Optional[int] = None
```

**Validierung in `_verify_args()` hinzufügen:**
- `ras_window_length` muss >= 1 sein (wenn gesetzt)
- `ras_max_num_repeat` muss >= 1 sein (wenn gesetzt)
- Beide müssen zusammen gesetzt werden oder beide None

---

### 1.3 Sampling Metadata: V1 Engine

**Datei:** `vllm/v1/sample/metadata.py`

**Änderungen:**
- `ras_window_length` und `ras_max_num_repeat` als Felder hinzufügen
- Diese Werte aus `SamplingParams` übernehmen

---

### 1.4 Sampler: RAS-Logik implementieren

**Datei:** `vllm/v1/sample/sampler.py`

**Neue Methode `apply_ras_penalty()`:**

```python
def apply_ras_penalty(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """
    Repetition Aware Sampling: Bestraft Tokens die zu oft
    im letzten Fenster vorkommen.

    Algorithmus:
    1. Für jede Sequenz: Hole die letzten `window_length` Tokens
    2. Zähle Vorkommen jedes Tokens im Fenster
    3. Wenn count >= max_num_repeat: Setze logit auf -inf
    """
    if sampling_metadata.ras_window_length is None:
        return logits

    for i, output_tokens in enumerate(sampling_metadata.output_token_ids):
        window = output_tokens[-sampling_metadata.ras_window_length:]
        token_counts = Counter(window)
        for token_id, count in token_counts.items():
            if count >= sampling_metadata.ras_max_num_repeat:
                logits[i, token_id] = float('-inf')

    return logits
```

**In `forward()` einbinden:**
- Nach `apply_penalties()`, vor `sample()` aufrufen

---

## Teil 2: Seed-Fixing

### 2.1 Problem-Analyse

Der V1 Sampler verwendet `sampling_metadata.generators` für Random Sampling.
Diese Generators werden aus dem Seed erstellt, aber der Seed wird möglicherweise
nicht korrekt von der Audio Speech API durchgereicht.

### 2.2 Request-Ebene

**Datei:** `vllm/entrypoints/openai/protocol.py`

**`AudioSpeechRequest.to_sampling_params()`:**
```python
def to_sampling_params(self) -> SamplingParams:
    return SamplingParams.from_optional(
        # ... bestehende Parameter ...
        seed=self.seed,  # NEU: Seed durchreichen
    )
```

### 2.3 Generator-Initialisierung prüfen

**Datei:** `vllm/v1/worker/gpu_input_batch.py`

**Prüfen:**
- Wird der Seed korrekt verwendet um `torch.Generator` zu erstellen?
- Wird der Generator pro Request isoliert?

**Erwartetes Verhalten:**
```python
if sampling_params.seed is not None:
    generator = torch.Generator(device=device)
    generator.manual_seed(sampling_params.seed)
```

### 2.4 Debugging-Schritte

1. Logging hinzufügen um zu prüfen ob Seed ankommt
2. Prüfen ob `SamplingType.RANDOM_SEED` korrekt gesetzt wird
3. Prüfen ob Generator korrekt initialisiert wird

---

## Teil 3: Serving Audio anpassen

**Datei:** `vllm/entrypoints/bosonai/serving_audio.py`

### 3.1 `create_audio_speech_stream()`

Keine Änderungen nötig - nutzt bereits `request.to_sampling_params()`

### 3.2 Optional: Default-Werte

Falls gewünscht, können Default-Werte für RAS in der Klasse definiert werden:

```python
class HiggsAudioServingAudio(OpenAIServing):
    DEFAULT_RAS_WINDOW_LENGTH = 7
    DEFAULT_RAS_MAX_NUM_REPEAT = 2
```

---

## Dateien-Übersicht

| Datei | Änderung |
|-------|----------|
| `vllm/entrypoints/openai/protocol.py` | RAS + Seed Felder in `AudioSpeechRequest` |
| `vllm/sampling_params.py` | RAS Felder in `SamplingParams` |
| `vllm/v1/sample/metadata.py` | RAS Felder in `SamplingMetadata` |
| `vllm/v1/sample/sampler.py` | `apply_ras_penalty()` Methode |
| `vllm/v1/worker/gpu_input_batch.py` | Seed-Handling prüfen/fixen |

---

## Test-Plan

### Unit Tests
- [ ] `SamplingParams` mit RAS-Parametern validieren
- [ ] RAS-Penalty-Logik mit bekannten Token-Sequenzen testen
- [ ] Seed-Reproduzierbarkeit testen (gleicher Seed = gleiche Ausgabe)

### Integration Tests
- [ ] Audio Speech API mit RAS-Parametern aufrufen
- [ ] Vergleich: Mit/Ohne RAS bei repetitivem Input
- [ ] Seed-Test: Zwei Requests mit gleichem Seed vergleichen

### Beispiel-Request
```python
response = client.audio.speech.create(
    model="higgs-audio",
    input="Hello, this is a test.",
    voice="alloy",
    seed=42,
    extra_body={
        "ras_window_length": 7,
        "ras_max_num_repeat": 2,
    }
)
```

---

---

## Teil 4: `/v1/audio/voices` Endpoint

### 4.1 Neuer Endpoint

**Datei:** `vllm/entrypoints/bosonai/api_server.py`

**Neuer Route hinzufügen:**
```python
@router.get("/v1/audio/voices")
async def list_voices(raw_request: Request):
    """List available voice presets (OpenAI-compatible)"""
    voice_presets = raw_request.app.state.voice_presets
    voices = []

    for voice_id, config in voice_presets.items():
        # Sprache aus voice_id ableiten (z.B. voice_de, zh_voice)
        language = detect_language_from_voice_id(voice_id)

        voices.append({
            "voice_id": voice_id,
            "name": voice_id.replace("_", " ").title(),
            "description": config.get("transcript", "")[:100],
            "labels": {"language": language}
        })

    return {"voices": voices}
```

### 4.2 Hilfsfunktion für Spracherkennung

```python
def detect_language_from_voice_id(voice_id: str) -> str:
    """Detect language from voice ID naming convention."""
    if voice_id.startswith("zh_"):
        return "chinese"

    suffix_map = {
        "_de": "german",
        "_es": "spanish",
        "_fr": "french",
        "_it": "italian",
        "_pt": "portuguese",
        "_ja": "japanese",
        "_ko": "korean",
    }

    for suffix, lang in suffix_map.items():
        if voice_id.endswith(suffix):
            return lang

    return "english"  # Default
```

---

## Teil 5: Usage Headers in Response

### 5.1 Problem

Aktuell werden keine Token-Statistiken in der Audio Speech Response zurückgegeben.
Diese sind wichtig für Monitoring und Billing.

### 5.2 Änderungen in `serving_audio.py`

**Datei:** `vllm/entrypoints/bosonai/serving_audio.py`

**Token-Zählung während Streaming:**
```python
async def audio_speech_stream_generator(...):
    prompt_tokens = 0
    completion_tokens = 0
    audio_tokens = 0

    async for res in result_generator:
        # Zähle Tokens
        if res.prompt_token_ids:
            prompt_tokens = len(res.prompt_token_ids)
        completion_tokens += len(output.token_ids)
        if output.mm_token_ids is not None:
            audio_tokens += output.mm_token_ids.shape[0]

        # ... existing streaming logic ...

    # Am Ende: Usage-Info yielden (als JSON-Chunk oder Header)
```

### 5.3 Response Headers

**In `api_server.py` anpassen:**
```python
@router.post("/v1/audio/speech")
async def create_speech(...):
    # ...

    # Custom headers mit Usage-Info
    headers = {
        "X-Usage-Prompt-Tokens": str(usage["prompt_tokens"]),
        "X-Usage-Completion-Tokens": str(usage["completion_tokens"]),
        "X-Usage-Audio-Tokens": str(usage["audio_tokens"]),
        "X-Usage-Total-Tokens": str(usage["total_tokens"]),
    }

    return StreamingResponse(
        audio_stream,
        media_type="audio/pcm",
        headers=headers
    )
```

**Problem:** Bei Streaming sind die finalen Token-Counts erst am Ende bekannt.

**Lösung:**
- Option A: Trailer-Headers (HTTP/2) - nicht überall unterstützt
- Option B: Finaler JSON-Chunk mit Usage-Info
- Option C: Separate `/v1/audio/speech/usage` Query nach Completion

**Empfehlung:** Option B - Finaler Chunk mit Usage-Info für Streaming, Headers für Non-Streaming.

---

## Teil 6: Speed Parameter (Sprechgeschwindigkeit)

### 6.1 Übersicht

Der `speed` Parameter ermöglicht schnelleres/langsameres Sprechen:
- `speed=0.5` → Halbe Geschwindigkeit (langsamer)
- `speed=1.0` → Normal (default)
- `speed=2.0` → Doppelte Geschwindigkeit (schneller)

### 6.2 Implementierung via Audio Time-Stretching

**Ansatz:** Post-Processing des generierten Audios mit `librosa.effects.time_stretch`

**Vorteile:**
- Einfach zu implementieren
- Pitch bleibt erhalten (kein Chipmunk-Effekt)
- Keine Modelländerungen nötig

**Nachteile:**
- Leichter Qualitätsverlust bei extremen Werten
- Zusätzliche Latenz

### 6.3 Änderungen

**Datei:** `vllm/entrypoints/openai/protocol.py`

```python
class AudioSpeechRequest(OpenAIBaseModel):
    # ... existing fields ...

    speed: float = 1.0
    """ Speech speed multiplier (0.25-4.0, default 1.0) """
```

**Datei:** `vllm/entrypoints/bosonai/serving_audio.py`

```python
import librosa

def apply_speed(audio: np.ndarray, speed: float, sr: int) -> np.ndarray:
    """
    Apply speed change to audio using time-stretching.

    Args:
        audio: Audio waveform as numpy array
        speed: Speed multiplier (0.25-4.0)
        sr: Sample rate

    Returns:
        Time-stretched audio
    """
    if speed == 1.0:
        return audio

    # Clamp speed to safe range
    speed = max(0.25, min(4.0, speed))

    # librosa time_stretch: rate > 1 = faster, rate < 1 = slower
    return librosa.effects.time_stretch(audio, rate=speed)
```

**In `audio_speech_stream_generator()` einbinden:**
```python
if audio_chunk is not None:
    # Apply speed if not 1.0
    if request.speed != 1.0:
        audio_chunk = apply_speed(
            audio_chunk,
            request.speed,
            self.audio_tokenizer.sampling_rate
        )

    output_audio, prev_resampled_audio = self._maybe_upsample_audio(...)
    yield output_audio
```

### 6.4 Validierung

**In `AudioSpeechRequest` oder `to_sampling_params()`:**
```python
def validate_speed(self):
    if self.speed < 0.25 or self.speed > 4.0:
        raise ValueError(f"speed must be between 0.25 and 4.0, got {self.speed}")
```

### 6.5 Streaming-Überlegungen

Bei Streaming muss beachtet werden:
- Time-Stretching ändert die Audio-Länge
- Chunk-Boundaries könnten verschoben werden
- Crossfade muss angepasst werden

**Lösung:** Speed nur auf finale Chunks anwenden, nicht während des Streamings.
Oder: Speed-Faktor in Chunk-Size-Berechnung einbeziehen.

---

## Dateien-Übersicht (Aktualisiert)

| Datei | Änderung |
|-------|----------|
| `vllm/entrypoints/openai/protocol.py` | RAS + Seed + Speed Felder |
| `vllm/sampling_params.py` | RAS Felder in `SamplingParams` |
| `vllm/v1/sample/metadata.py` | RAS Felder in `SamplingMetadata` |
| `vllm/v1/sample/sampler.py` | `apply_ras_penalty()` Methode |
| `vllm/v1/worker/gpu_input_batch.py` | Seed-Handling prüfen/fixen |
| `vllm/entrypoints/bosonai/api_server.py` | `/v1/audio/voices` Endpoint, Usage Headers |
| `vllm/entrypoints/bosonai/serving_audio.py` | Speed-Funktion, Token-Zählung |

---

## Test-Plan (Erweitert)

### Unit Tests
- [ ] `SamplingParams` mit RAS-Parametern validieren
- [ ] RAS-Penalty-Logik mit bekannten Token-Sequenzen testen
- [ ] Seed-Reproduzierbarkeit testen (gleicher Seed = gleiche Ausgabe)
- [ ] Speed-Funktion mit verschiedenen Werten testen
- [ ] Usage-Zählung validieren

### Integration Tests
- [ ] Audio Speech API mit RAS-Parametern aufrufen
- [ ] Vergleich: Mit/Ohne RAS bei repetitivem Input
- [ ] Seed-Test: Zwei Requests mit gleichem Seed vergleichen
- [ ] `/v1/audio/voices` Endpoint testen
- [ ] Usage Headers in Response prüfen
- [ ] Speed-Parameter testen (0.5, 1.0, 2.0)

### Beispiel-Requests

**Speech mit allen Features:**
```python
response = client.audio.speech.create(
    model="higgs-audio",
    input="Hello, this is a test.",
    voice="alloy",
    speed=1.2,  # Etwas schneller
    extra_body={
        "seed": 42,
        "ras_window_length": 7,
        "ras_max_num_repeat": 2,
    }
)

# Usage aus Headers lesen
print(response.headers.get("X-Usage-Total-Tokens"))
```

**Voices auflisten:**
```bash
curl http://localhost:8000/v1/audio/voices
```

**Response:**
```json
{
  "voices": [
    {
      "voice_id": "belinda",
      "name": "Belinda",
      "description": "English female voice...",
      "labels": {"language": "english"}
    },
    {
      "voice_id": "ingrid_de",
      "name": "Ingrid De",
      "description": "German female voice...",
      "labels": {"language": "german"}
    }
  ]
}
```

---

## Geschätzter Aufwand (Aktualisiert)

| Task | Aufwand |
|------|---------|
| RAS in Protocol/SamplingParams | ~30 min |
| RAS in Metadata/Sampler | ~1 h |
| Seed-Fixing | ~30 min - 1 h |
| `/v1/audio/voices` Endpoint | ~30 min |
| Usage Headers | ~1 h |
| Speed Parameter | ~1-2 h |
| Tests | ~1-2 h |
| **Gesamt** | **~6-8 h** |
