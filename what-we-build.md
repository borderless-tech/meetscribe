# meetscribe — Bootstrap-Spec

Handoff-Dokument für Claude Code. Ziel: dieses Projekt von Null auf lauffähig bringen.

---

## 1. Was gebaut wird

Ein lokales, plattformübergreifendes (Linux + macOS) CLI-Tool, das ein Meeting aufnimmt
(Mikrofon **und** System-Audio) und daraus drei Artefakte erzeugt:

- `transcript.json` — Text mit Sprecher-Annotation und Timestamps
- `embeddings.npz` — Sprecher-Embeddings (Vektoren) pro Turn und pro Cluster
- `meta.json` — Modellnamen, Versionen, Vektordimension

Die Embeddings wandern später in eine pgvector-Instanz, um Sprecher **über Meetings hinweg**
wiederzuerkennen. Deshalb sind die rohen Vektoren ein First-Class-Output und kein Nebenprodukt.

Alles läuft offline auf CPU. Kein Cloud-Call, kein PyTorch, kein CUDA.

### Ziel-UX

```bash
nix run github:<user>/meetscribe#doctor   # einmalig: Audio-Setup prüfen
nix run github:<user>/meetscribe          # aufnehmen (Ctrl-C stoppt) + verarbeiten
nix run github:<user>/meetscribe -- process path/to/audio.wav   # nur verarbeiten
```

Kein Clone, kein venv, kein pip, kein manueller Modell-Download.

---

## 2. Architektur-Entscheidungen (bitte nicht wegoptimieren)

Diese Punkte sind bewusst so gewählt. Wenn du meinst, es ginge einfacher, lies erst die
Begründung — sie sind alle aus konkreten Problemen entstanden.

### 2.1 Dual-Track statt gemischter Aufnahme

Mikrofon und System-Output werden als **zwei getrennte Mono-Files** aufgenommen, nicht gemischt
und nicht als Stereo-L/R.

- `mic.wav` ist definitionsgemäß der User. Braucht **keine** Diarization, kein Clustering,
  ist 100 % korrekt gelabelt (`speaker: "me"`).
- `system.wav` enthält alle anderen. Nur hier läuft Diarization.

Das halbiert den Rechenaufwand und eliminiert die häufigste Fehlerquelle (eigener Voice-Cluster
kollidiert mit einem Gesprächspartner). Kein Stereo, weil ASR-Pipelines Stereo zu Mono
downmixen und die Kanalinfo dabei verloren geht.

### 2.2 sherpa-onnx statt pyannote/PyTorch

Begründung: ein Runtime-Dependency (ONNX Runtime, kommt im Wheel mit), CPU-only, prebuilt Wheels
für alle vier relevanten Targets, kein HF-Token, kein CUDA-vs-MPS-Codepfad. Identischer Code auf
macOS und Linux.

### 2.3 Parakeet TDT **v3**, nicht v2

`parakeet-tdt-0.6b-v2` ist English-only. Es muss **v3** sein (multilingual, inkl. Deutsch) —
die Meetings sind gemischt deutsch/englisch.

Zweiter Grund für Parakeet statt Whisper unter sherpa-onnx: TDT liefert native Token-Timestamps.
Whisper tut das unter sherpa-onnx nur schlecht. Timestamps sind hier nicht optional (Merge der
zwei Tracks + Word-zu-Speaker-Zuordnung hängen daran).

### 2.4 VAD ist Pflicht, nicht Bonus

Der Offline-Recognizer verarbeitet ein File am Stück. Bei 60-Minuten-Meetings gibt das Speicher-
und Qualitätsprobleme. Silero-VAD davor, Chunks von ~15–30 s an den Recognizer, Offsets
zurückrechnen.

### 2.5 Zwei Pässe für Diarization und Embeddings

Die Diarization-API von sherpa-onnx gibt **nur** Segmentgrenzen + Cluster-Label zurück. Die
intern berechneten Embeddings kommen nicht durch die API raus. Also ein zweiter Pass mit dem
Speaker-Embedding-Extractor über dieselben Segmente. Gleiches Modell, doppelte Rechnung — auf CPU
so schnell, dass es egal ist.

### 2.6 Embeddings auch für `mic.wav`

Der Extractor läuft auch über die Mic-Segmente. Damit entsteht ein sauber gelabeltes Eigenprofil
ohne Clustering-Risiko — nützlich als Kalibrier-Referenz für die Similarity-Schwellen.

### 2.7 Cluster-Centroid: Audio konkatenieren, nicht Vektoren mitteln

Für das Cluster-Profil alle Sprachabschnitte eines Clusters aneinanderhängen und daraus **einen**
Embedding ziehen. Deutlich stabiler als der Mittelwert über viele kurze Turns.

---

## 3. Repo-Layout

```
meetscribe/
├── flake.nix
├── flake.lock
├── pyproject.toml
├── uv.lock
├── nix/
│   ├── models.nix          # Modelle als fetchurl, gepinnt
│   └── package.nix         # Wrapper-Derivation
├── src/meetscribe/
│   ├── __init__.py
│   ├── cli.py              # argparse: record | process | doctor
│   ├── doctor.py           # Preflight-Checks
│   ├── record.py           # Plattform-Switch (der EINZIGE plattformspezifische Teil)
│   ├── audio.py            # Resample/Konvertierung via ffmpeg
│   ├── vad.py
│   ├── asr.py
│   ├── diarize.py
│   ├── embed.py
│   ├── align.py            # Word → Speaker Overlap-Zuordnung
│   ├── merge.py            # Zwei Tracks nach Timestamp interleaven
│   └── output.py           # transcript.json / embeddings.npz / meta.json
├── tests/
└── .github/workflows/build.yml
```

---

## 4. Pipeline

```
record.sh / record.py
  └→ raw/mic.wav, raw/system.wav        (48k, nativ)
       └→ ffmpeg -ar 16000 -ac 1
            └→ work/mic16.wav, work/system16.wav
                 │
    ┌────────────┴────────────┐
    │                         │
  mic16.wav               system16.wav
    │                         │
  VAD → chunks            Diarization → [(start, end, spk_N)]
    │                         │
  ASR (Parakeet)          VAD → chunks → ASR
    │                         │
  segments(speaker="me")  segments + word-timestamps
    │                         │
    │                     align: words → spk (max overlap)
    │                         │
    │                     Embeddings: pro Turn + pro Cluster
    └──────────┬──────────────┘
          merge by timestamp
               │
    transcript.json · embeddings.npz · meta.json
```

### 4.1 Word → Speaker Zuordnung (`align.py`)

ASR-Segmente und Diarization-Segmente haben unterschiedliche Grenzen, die passen nie zusammen.
Standardverfahren (macht WhisperX intern genauso):

1. Für jedes Wort `[w_start, w_end]` die zeitliche Überlappung mit jedem Diarization-Segment
   berechnen.
2. Segment mit größter Überlappung gewinnt.
3. Aufeinanderfolgende Wörter mit gleichem Label zu Utterances zusammenfassen.

~40 Zeilen. Bitte keine Heuristik-Erfindungen, das Verfahren ist erprobt.

### 4.2 Merge (`merge.py`)

Beide Utterance-Listen in eine Liste, nach `start` sortieren. Mic-Utterances bekommen hart
`speaker: "me"` und werden von keinem Clustering angefasst.

---

## 5. Aufnahme (`record.py`) — der einzige plattformspezifische Teil

### macOS

Voraussetzung (kann Nix **nicht** installieren, siehe §8):
- BlackHole 2ch (`brew install blackhole-2ch`)
- Multi-Output Device (BlackHole + Kopfhörer) als System-Output
- Aggregate Device (BlackHole + Mic), benannt `meetscribe`

Aufnahme via `ffmpeg -f avfoundation` gegen das Aggregate Device, Kanäle per `pan`-Filter in zwei
Files splitten.

Das Aggregate Device ist wichtig, weil es Drift-Korrektur zwischen den beiden Clocks macht. Zwei
separate ffmpeg-Inputs tun das nicht.

**Device-Indizes niemals hardcoden.** Per `ffmpeg -f avfoundation -list_devices true -i ""`
auflösen und nach Namen matchen — die Indizes verschieben sich.

### Linux

`ffmpeg -f pulse` zweimal: einmal Mic-Source, einmal `<sink>.monitor`. Kein Setup nötig,
PipeWire kann das out of the box. Alternativ `pw-record`.

### Beide

- `trap` auf SIGINT, ffmpeg ein `q` auf stdin schicken statt hartem Kill — sonst gelegentlich
  kaputte WAV-Header.
- Nach der Aufnahme RMS beider Files prüfen und warnen, wenn ein Track stumm ist.

---

## 6. Output-Formate

```jsonc
// transcript.json
{
  "meeting_id": "2026-08-02T14-30-00",
  "duration_s": 3412.5,
  "segments": [
    {
      "start": 12.4, "end": 18.9,
      "speaker": "me",              // oder "spk_0", "spk_1", ...
      "track": "mic",               // oder "system"
      "text": "...",
      "words": [{ "w": "...", "start": 12.4, "end": 12.7 }]
    }
  ]
}
```

```
# embeddings.npz
turn_ids     : (N,)      str
turn_vectors : (N, 512)  float32
turn_speakers: (N,)      str
cluster_ids  : (K,)      str
cluster_vectors: (K, 512) float32
```

```jsonc
// meta.json  — NICHT optional
{
  "embedding_model": "cam++",
  "embedding_model_sha256": "...",
  "embedding_dim": 512,
  "asr_model": "parakeet-tdt-0.6b-v3",
  "segmentation_model": "pyannote-segmentation-3.0-onnx",
  "sample_rate": 16000,
  "meetscribe_version": "0.1.0"
}
```

**Warum `meta.json` kritisch ist:** Vektoren aus verschiedenen Modellen sind nicht vergleichbar.
Ohne Modellname + Dimension in der Metadata ist die pgvector-Historie beim ersten Modellwechsel
stumm kaputt. Diese Felder gehören später auch in die DB-Tabelle.

---

## 7. Nix / Flake

### 7.1 uv2nix für die Python-Deps

sherpa-onnx ist nicht in nixpkgs, hat aber Wheels für `macosx_11_0_arm64`,
`macosx_10_15_x86_64`, `manylinux2014_x86_64`, `manylinux2014_aarch64`.

Deshalb: `pyproject.toml` + `uv lock` → uv2nix baut daraus reproduzierbar einen Python-Env.

Inputs: `nixpkgs`, `flake-utils`, `pyproject-nix`, `uv2nix`, `pyproject-build-systems`
(mit `follows` für nixpkgs/pyproject-nix überall).

```nix
workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
overlay   = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

pythonSet = (pkgs.callPackage pyproject-nix.build.packages { python = pkgs.python312; })
  .overrideScope (lib.composeManyExtensions [
    pyproject-build-systems.overlays.default
    overlay
    (final: prev: {
      sherpa-onnx-core = prev.sherpa-onnx-core.overrideAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or [])
          ++ lib.optional stdenv.isLinux pkgs.autoPatchelfHook;
        buildInputs = (old.buildInputs or []) ++ [ pkgs.stdenv.cc.cc.lib ];
      });
    })
  ]);

venv = pythonSet.mkVirtualEnv "meetscribe-env" workspace.deps.default;
```

Zwei Fallstricke:
- `sourcePreference = "wheel"` ist **Pflicht**. Source-Build zieht CMake + halbe ONNX-Runtime rein.
- sherpa-onnx ist auf mehrere Distributionen gesplittet: `sherpa_onnx`, `sherpa_onnx_core`,
  `sherpa_onnx_bin`. **Alle drei** müssen im Lock landen, sonst fehlt zur Laufzeit die native Lib.

Python-Deps ansonsten: `numpy`. Mehr nicht.

### 7.2 Modelle als `fetchurl`

Modelle werden Teil der Flake, kein Download-Skript:

```nix
# nix/models.nix
runCommand "meetscribe-models" {} ''
  mkdir -p $out/{asr,seg,spk,vad}
  tar xf ${asr}  -C $out/asr --strip-components=1
  cp ${segmentation} $out/seg/model.onnx
  cp ${campplus}     $out/spk/model.onnx
  cp ${silero}       $out/vad/silero_vad.onnx
''
```

Damit sind die Hashes gepinnt und ein stiller Modellwechsel (der alle bestehenden Embeddings
unvergleichbar machen würde) ist strukturell ausgeschlossen.

Parakeet v3 int8 ist ein paar hundert MB und landet komplett in der Closure. Erstmal so lassen.
Falls es stört: ASR-Modell als separates Output rausziehen und lazy nach `$XDG_CACHE_HOME`
laden; die kleinen Speaker-Modelle (~45 MB) bleiben gepinnt.

### 7.3 Wrapper

```nix
packages.default = pkgs.stdenv.mkDerivation {
  pname = "meetscribe";
  src = ./.;
  nativeBuildInputs = [ pkgs.makeWrapper ];
  installPhase = ''
    install -Dm755 bin/meetscribe $out/bin/meetscribe
    wrapProgram $out/bin/meetscribe \
      --prefix PATH : ${lib.makeBinPath [ venv pkgs.ffmpeg-headless ]} \
      --set MEETSCRIBE_MODELS ${models}
  '';
};
apps.default = { type = "app"; program = "…/bin/meetscribe"; };
```

ffmpeg kommt aus dem Store. Der User braucht außer Nix **nichts** vorinstalliert (Ausnahme:
BlackHole auf macOS).

Zusätzlich exportieren: `overlays.default`, `homeManagerModules.default` (installiert
`meetscribe`, optional launchd/systemd-User-Service für Auto-Processing).

### 7.4 Systems

`x86_64-linux`, `aarch64-linux`, `aarch64-darwin`, `x86_64-darwin`.

---

## 8. `doctor` — halbe UX, kein Nice-to-have

Auf macOS kann Nix zwei Dinge prinzipiell nicht: BlackHole installieren (CoreAudio-HAL-Plugin,
gehört nach `/Library/Audio/Plug-Ins/HAL`, braucht root und einen `coreaudiod`-Restart) und das
Aggregate Device anlegen. Deshalb muss `doctor` den User da sauber durchführen.

```
$ nix run github:<user>/meetscribe#doctor
✓ ffmpeg 7.x
✓ Modelle (parakeet-tdt-0.6b-v3, cam++, seg-3.0)
✗ BlackHole nicht gefunden
    → brew install blackhole-2ch
✗ Aggregate Device "meetscribe" nicht gefunden
    → Audio MIDI Setup öffnen, BlackHole 2ch + Mic kombinieren, so benennen
✓ Mikrofon-Zugriff (1s Testaufnahme, RMS > 0)
```

Der letzte Check ist der wichtigste: **ohne TCC-Freigabe nimmt ffmpeg stillschweigend Stille auf**
— kein Fehler, kein Exit-Code ≠ 0. Ohne diesen Check merkt man es erst nach dem Meeting.

(Nebenbei: TCC vergibt die Berechtigung an das Terminal, nicht an den Store-Pfad. Der Store-Pfad
ändert sich bei Updates, das Prompt kommt trotzdem nur einmal.)

Auf Linux ist `doctor` trivial: läuft PipeWire, ist eine Monitor-Source da.

---

## 9. CI

GitHub Actions Matrix: `ubuntu-latest` + `macos-14` (arm64). Build → Push nach Cachix.

```nix
nixConfig = {
  extra-substituters = [ "https://meetscribe.cachix.org" ];
  extra-trusted-public-keys = [ "meetscribe.cachix.org-1:…" ];
};
```

Ohne Cachix funktioniert alles trotzdem, der erste `nix run` dauert nur länger.

---

## 10. Vor dem Bauen verifizieren

Diese Punkte sind aus dem Gedächtnis geschrieben und **müssen** gegen die aktuelle Doku geprüft
werden, bevor du Code schreibst:

1. **Exakte sherpa-onnx-Python-API-Namen.** Die Offline-Diarization-API ist relativ neu und hat
   sich zwischen Releases bewegt. Klassennamen für Diarization, Speaker-Embedding-Extractor,
   VAD und Offline-Recognizer gegen die Doku der Version prüfen, die du pinnst.
   → https://k2-fsa.github.io/sherpa/onnx/python/
2. **Exakte Modell-URLs und -Namen** für Parakeet TDT v3, CAM++, pyannote-segmentation-3.0-ONNX,
   Silero VAD. Nicht raten — aus dem sherpa-onnx-Model-Index holen.
3. **Aktuelle sherpa-onnx-Version** und ob das Package-Splitting (`_core`/`_bin`) in dieser
   Version so aussieht.
4. **uv2nix-API.** Bewegt sich ebenfalls; das Snippet oben ist ein Skelett, kein Copy-Paste.

Version in `pyproject.toml` festnageln, nicht `>=`.

---

## 11. Reihenfolge

1. **`record` zuerst.** Plattform-Switch, zwei Tracks, Ctrl-C-Handling, RMS-Check. Verifizieren,
   dass beide Tracks sauber und synchron sind (reinhören). Das ist der Teil, der schiefgeht —
   vor allem ML-Kram erledigen.
2. `doctor`.
3. Flake + uv2nix, so dass `nix run … -- doctor` durchläuft. Beide Plattformen.
4. Modelle als `fetchurl`, Pipeline-Stub der die Modelle lädt und wieder beendet.
5. ASR-Pfad: VAD → Parakeet → Segmente mit Word-Timestamps. Erst nur `mic.wav`, ohne Speaker.
6. Diarization + Embeddings auf `system.wav`.
7. `align` + `merge` + Output-Schreiber.
8. CI + Cachix.

pgvector-Ingest und Profil-Matching sind **nicht** Teil dieses Projekts — separates Tool, das
`embeddings.npz` + `meta.json` konsumiert.

---

## 12. Bekannte Fallstricke

| Problem | Umgang |
|---|---|
| Echo/Bleed: ohne Kopfhörer landet die Gegenseite auch im Mic-Track | v1: Kopfhörer-Pflicht, in README dokumentieren. Später ggf. WebRTC AEC3 als Sidecar. |
| Segmente < ~0.8 s liefern unbrauchbare Embeddings | rausfiltern statt clustern |
| Unbekannte Sprecheranzahl | Agglomerative Clustering mit Schwellwert, **kein** fixes K |
| avfoundation-Device-Indizes verschieben sich | nach Namen matchen |
| ffmpeg nimmt ohne TCC still Stille auf | `doctor`-RMS-Check |
| WAV-Header kaputt bei hartem Kill | `q` auf stdin statt SIGKILL |
| Clock-Drift macOS | über Aggregate Device gelöst |

---

## 13. Später (nicht v1)

- Core Audio Taps statt BlackHole (macOS 14.2+) über einen Swift-Helper — eliminiert den einzigen
  manuellen Installationsschritt. Swift unter Nix auf darwin ist aber ein eigenes Projekt.
- Realtime/Streaming statt Batch.
- pgvector-Ingest-Tool.
