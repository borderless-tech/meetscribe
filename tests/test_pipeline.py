"""Pipeline orchestration: input resolution + two-track wiring (fast, with fake stages).

The heavy real-model end-to-end smoke lives in tests/test_e2e.py (opt-in, needs MEETSCRIBE_MODELS).
"""

import wave

import numpy as np
import pytest

from meetscribe.asr import RawResult
from meetscribe.pipeline import Components, process, resolve_inputs
from meetscribe.vad import Chunk


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    """No test may read the real home: the config is a per-test tmp file (missing by
    default → all defaults) and the XDG dirs point into tmp_path (run() also reads the
    glossary under XDG_CONFIG_HOME). Tests that need a config layer write
    ``tmp_path / "config.toml"``. The STT env vars are scrubbed too — a developer
    shell exporting STT_BACKEND=deepgram must not send run() tests down the
    unstubbed remote path."""
    monkeypatch.setenv("MEETSCRIBE_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    for var in ("STT_BACKEND", "STT_LANGUAGE", "DEEPGRAM_API_KEY",
                "MEETSCRIBE_BK_URL", "MEETSCRIBE_BK_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def _write_wav(path, seconds=1.0, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.zeros(int(seconds * rate), dtype="<i2").tobytes())


# ---- resolve_inputs -----------------------------------------------------------------

def test_resolve_inputs_directory_with_raw(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_wav(raw / "mic.wav")
    _write_wav(raw / "system.wav")
    mic, system = resolve_inputs(str(tmp_path))
    assert mic.endswith("raw/mic.wav") and system.endswith("raw/system.wav")


def test_resolve_inputs_single_file_is_system(tmp_path):
    f = tmp_path / "meeting.wav"
    _write_wav(f)
    mic, system = resolve_inputs(str(f))
    assert mic is None and system.endswith("meeting.wav")


def test_resolve_inputs_mic_only(tmp_path):
    _write_wav(tmp_path / "mic.wav")
    mic, system = resolve_inputs(str(tmp_path))
    assert mic.endswith("mic.wav") and system is None


# ---- process (fake stages) ----------------------------------------------------------

MARK = "▁"


class FakeVad:
    def chunks(self, samples):
        return [Chunk(0.0, np.asarray(samples, dtype=np.float32), 16000)]


class FakeRecognizer:
    def recognize(self, samples):
        # spans 0.0–1.0 s so the utterance survives the 0.8 s short-segment filter
        return RawResult("hello world", [f"{MARK}hello", f"{MARK}world"], [0.0, 0.6], [0.3, 0.4])


class FakeDiarizer:
    def segments(self, samples, on_progress=None):
        from dataclasses import dataclass

        @dataclass
        class S:
            start: float
            end: float
            speaker: int

        if on_progress is not None:
            on_progress(1, 1)
        return [S(0.0, 1.0, 0)]


class FakeEmbedder:
    dim = 192

    def embed(self, samples):
        return np.ones(192, dtype=np.float32)


def _components():
    return Components(FakeVad(), FakeRecognizer(), FakeDiarizer(), FakeEmbedder())


def test_process_both_tracks(tmp_path):
    _write_wav(tmp_path / "mic.wav")
    _write_wav(tmp_path / "system.wav")
    res = process(str(tmp_path / "mic.wav"), str(tmp_path / "system.wav"), _components())

    speakers = {u.speaker for u in res.utterances}
    tracks = {u.track for u in res.utterances}
    assert "me" in speakers and "spk_0" in speakers
    assert tracks == {"mic", "system"}
    assert res.dim == 192
    # turns from both tracks, ids prefixed & unique
    ids = [t[0] for t in res.turns]
    assert any(i.startswith("mic_") for i in ids)
    assert any(i.startswith("sys_") for i in ids)
    assert len(ids) == len(set(ids))
    # a "me" centroid and a spk_0 centroid
    cluster_ids = {c[0] for c in res.clusters}
    assert "me" in cluster_ids and "spk_0" in cluster_ids
    assert res.duration_s == 1.0


class ScriptedDiarizer:
    """Diarizer returning a fixed list of (start, end, int_speaker) segments."""

    def __init__(self, segs):
        self._segs = segs

    def segments(self, samples, on_progress=None):
        from dataclasses import dataclass

        @dataclass
        class S:
            start: float
            end: float
            speaker: int

        return [S(*t) for t in self._segs]


class ScriptedRecognizer:
    """Recognizer returning fixed word timestamps."""

    def __init__(self, tokens, starts, durations):
        self._raw = RawResult(
            " ".join(tokens), [f"{MARK}{t}" for t in tokens], starts, durations
        )

    def recognize(self, samples):
        return self._raw


def _embedding_speakers(res):
    return {t[2] for t in res.turns} | {c[0] for c in res.clusters}


def test_wordless_diar_cluster_gets_no_embeddings(tmp_path):
    """A diarization cluster that won no words must not appear in the embeddings.

    Regression: meeting 2026-08-04 produced turn/cluster vectors for spk_N labels
    with no transcript segment, which upload sinks reject."""
    _write_wav(tmp_path / "system.wav", seconds=3.0)
    # words span 0.0–1.0 s → all land on spk_0; spk_1 (2.0–3.0, ≥0.8 s) gets none
    comps = Components(
        FakeVad(),
        FakeRecognizer(),
        ScriptedDiarizer([(0.0, 1.0, 0), (2.0, 3.0, 1)]),
        FakeEmbedder(),
    )
    res = process(None, str(tmp_path / "system.wav"), comps)

    transcript_speakers = {u.speaker for u in res.utterances}
    assert "spk_1" not in transcript_speakers  # sanity: no words → not in transcript
    assert _embedding_speakers(res) == transcript_speakers


def test_short_spoken_cluster_still_gets_centroid(tmp_path):
    """A speaker whose diar segments are all <0.8 s but who won words must still
    get a cluster centroid (concatenation supplies the audio); no turn vectors."""
    _write_wav(tmp_path / "system.wav", seconds=1.0)
    comps = Components(
        FakeVad(),
        FakeRecognizer(),
        ScriptedDiarizer([(0.0, 0.5, 0)]),
        FakeEmbedder(),
    )
    res = process(None, str(tmp_path / "system.wav"), comps)

    assert "spk_0" in {u.speaker for u in res.utterances}  # sanity: it spoke
    assert "spk_0" in {c[0] for c in res.clusters}
    assert all(t[2] != "spk_0" for t in res.turns)  # sub-0.8 s: no turn vectors
    assert _embedding_speakers(res) == {u.speaker for u in res.utterances}


def test_short_mic_utterance_still_gets_me_centroid(tmp_path):
    """Mic track: a <0.8 s utterance is in the transcript, so 'me' needs a centroid."""
    _write_wav(tmp_path / "mic.wav", seconds=1.0)
    comps = Components(
        FakeVad(),
        ScriptedRecognizer(["hi"], [0.0], [0.5]),
        ScriptedDiarizer([]),
        FakeEmbedder(),
    )
    res = process(str(tmp_path / "mic.wav"), None, comps)

    assert {u.speaker for u in res.utterances} == {"me"}
    assert "me" in {c[0] for c in res.clusters}
    assert _embedding_speakers(res) == {"me"}


class RecordingReporter:
    def __init__(self):
        self.stages = []
        self.tracks = []
        self.advances = 0
        self.warns = []

    def stage(self, label):
        self.stages.append(label)
        import contextlib

        return contextlib.nullcontext()

    def track(self, label, total):
        self.tracks.append(label)
        outer = self

        class T:
            def advance(self, n=1):
                outer.advances += 1

            def __enter__(self):
                return self

            def __exit__(self, *e):
                pass

        return T()

    def info(self, msg):
        pass

    def warn(self, msg):
        self.warns.append(msg)

    def summary(self, s):
        pass


def test_process_reports_stages_and_asr_progress(tmp_path):
    _write_wav(tmp_path / "mic.wav")
    _write_wav(tmp_path / "system.wav")
    rep = RecordingReporter()
    process(str(tmp_path / "mic.wav"), str(tmp_path / "system.wav"), _components(), reporter=rep)
    assert any("transcribe" in s for s in rep.stages)
    assert any("diarize" in s for s in rep.stages)
    assert rep.advances >= 1  # ASR chunk progress advanced


def test_process_reports_diarize_and_embed_progress(tmp_path):
    # The long CPU stages (diarization, embedding) must drive progress bars, not
    # sit silent behind a spinner for minutes on a real meeting.
    _write_wav(tmp_path / "mic.wav")
    _write_wav(tmp_path / "system.wav")
    rep = RecordingReporter()
    process(str(tmp_path / "mic.wav"), str(tmp_path / "system.wav"), _components(), reporter=rep)

    assert any("diarization" in label for label in rep.tracks)
    assert any("embedding (mic)" in label for label in rep.tracks)
    assert any("embedding (system)" in label for label in rep.tracks)
    # ASR (2 chunks) + diarization percent + 2×(turn + centroid) embeds
    assert rep.advances >= 7


def test_process_warns_on_dropped_cluster(tmp_path):
    """A diarization cluster that wins no words vanishes from the transcript with no
    trace. When it does, process() must warn — otherwise `--speakers 4` silently
    yielding 2 transcript speakers looks like the count was ignored.

    Regression: meeting 2026-08-12 forced 3 clusters (spk_0/1/2); spk_0 caught 2 s of
    audio, won 0 words, and disappeared with no warning."""
    _write_wav(tmp_path / "system.wav", seconds=3.0)
    # words span 0.0–1.0 s → all land on spk_0; spk_1 (2.0–3.0) wins nothing → dropped
    comps = Components(
        FakeVad(),
        FakeRecognizer(),
        ScriptedDiarizer([(0.0, 1.0, 0), (2.0, 3.0, 1)]),
        FakeEmbedder(),
    )
    rep = RecordingReporter()
    res = process(None, str(tmp_path / "system.wav"), comps, reporter=rep)

    assert "spk_1" not in {u.speaker for u in res.utterances}  # sanity: dropped
    assert any("spk_1" in w for w in rep.warns), rep.warns


def test_process_no_warning_when_all_clusters_spoke(tmp_path):
    """No spurious warning when every diarized cluster wins words."""
    _write_wav(tmp_path / "system.wav")
    rep = RecordingReporter()
    process(None, str(tmp_path / "system.wav"), _components(), reporter=rep)
    assert rep.warns == []


class UpperCleaner:
    """Fake cleaner: uppercases each utterance's text (stashing raw_text), reports active."""

    model_info = {"name": "fake-llm", "sha256": "deadbeef"}

    def clean(self, utterances, glossary, reporter):
        from dataclasses import replace

        from meetscribe.cleanup import RepairResult

        out = [replace(u, text=u.text.upper(), raw_text=u.text) for u in utterances]
        return RepairResult(out, fixed=len(out), echoes=0, active=True)


def test_process_cleans_system_track_only(tmp_path):
    """Cleanup rewrites system-track text (the degraded downmix), sets raw_text to the
    original, and leaves the mic ('me') track untouched. Timings are never touched."""
    _write_wav(tmp_path / "mic.wav")
    _write_wav(tmp_path / "system.wav")
    comps = Components(FakeVad(), FakeRecognizer(), FakeDiarizer(), FakeEmbedder(), UpperCleaner())
    res = process(str(tmp_path / "mic.wav"), str(tmp_path / "system.wav"), comps)

    by_track = {u.track: u for u in res.utterances}
    assert by_track["system"].text == "HELLO WORLD"        # cleaned
    assert by_track["system"].raw_text == "hello world"     # original preserved
    assert by_track["mic"].text == "hello world"            # mic NOT cleaned
    # words verbatim on both tracks
    assert by_track["system"].words[0].w == "hello"
    assert res.cleaned is True
    assert res.cleanup_model == {"name": "fake-llm", "sha256": "deadbeef"}


def test_process_nullcleaner_leaves_text_and_marks_uncleaned(tmp_path):
    _write_wav(tmp_path / "system.wav")
    res = process(None, str(tmp_path / "system.wav"), _components())  # default NullCleaner
    assert res.utterances[0].text == "hello world"
    assert res.cleaned is False
    assert res.cleanup_model is None


def test_process_system_only(tmp_path):
    _write_wav(tmp_path / "system.wav")
    res = process(None, str(tmp_path / "system.wav"), _components())
    assert all(u.track == "system" for u in res.utterances)
    assert all(t[0].startswith("sys_") for t in res.turns)


def test_default_bundle_name_uses_meeting_id():
    from meetscribe.output import default_bundle_name

    assert default_bundle_name({"meeting_id": "abc123"}) == "meeting-abc123.mscribe"


def test_derive_window_from_started_at():
    from datetime import datetime, timezone

    from meetscribe.pipeline import _window

    start = datetime(2026, 8, 2, 14, 0, 0, tzinfo=timezone.utc)
    started, ended = _window(started_at=start, duration_s=60.0, mic_wav=None, system_wav=None)
    assert started == "2026-08-02T14:00:00+00:00"
    assert ended == "2026-08-02T14:01:00+00:00"


# ---- run --bundle (fake stages) -----------------------------------------------------

def _fake_models_dir(tmp_path):
    """A models dir just complete enough for run(): a spk model file to sha256."""
    spk = tmp_path / "models" / "spk"
    spk.mkdir(parents=True)
    (spk / "model.onnx").write_bytes(b"fake")
    return str(spk.parent)


def test_run_with_bundle_writes_mscribe(tmp_path, monkeypatch):
    from meetscribe import pipeline

    rec = tmp_path / "rec"
    (rec / "raw").mkdir(parents=True)
    _write_wav(rec / "raw" / "mic.wav")
    _write_wav(rec / "raw" / "system.wav")

    monkeypatch.setenv("MEETSCRIBE_MODELS", _fake_models_dir(tmp_path))
    monkeypatch.setattr(
        pipeline, "build_components", lambda models_dir, num_speakers=-1, cleanup=True: _components()
    )

    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out), bundle=True) == 0

    assert (out / f"meeting-{out.name}.mscribe").exists()
    # the loose dir is still produced alongside the bundle
    assert (out / "transcript.json").exists()
    assert (out / "embeddings.npz").exists()
    assert (out / "meta.json").exists()


def test_run_without_bundle_writes_no_mscribe(tmp_path, monkeypatch):
    # explicit bundle=False (the --no-bundle flag) always wins
    from meetscribe import pipeline

    rec = tmp_path / "rec"
    (rec / "raw").mkdir(parents=True)
    _write_wav(rec / "raw" / "system.wav")

    monkeypatch.setenv("MEETSCRIBE_MODELS", _fake_models_dir(tmp_path))
    monkeypatch.setattr(
        pipeline, "build_components", lambda models_dir, num_speakers=-1, cleanup=True: _components()
    )

    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out), bundle=False) == 0
    assert not list(out.glob("*.mscribe"))


def _bundle_run_env(tmp_path, monkeypatch):
    from meetscribe import pipeline

    rec = tmp_path / "rec"
    (rec / "raw").mkdir(parents=True)
    _write_wav(rec / "raw" / "system.wav")
    monkeypatch.setenv("MEETSCRIBE_MODELS", _fake_models_dir(tmp_path))
    monkeypatch.setattr(
        pipeline, "build_components", lambda models_dir, num_speakers=-1, cleanup=True: _components()
    )
    return rec


def test_run_bundle_none_defaults_on(tmp_path, monkeypatch):
    # bundle=None (flag not given) with no config → the built-in default (on).
    from meetscribe import pipeline

    rec = _bundle_run_env(tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out)) == 0
    assert (out / f"meeting-{out.name}.mscribe").exists()


def test_run_bundle_none_resolves_from_config(tmp_path, monkeypatch):
    # bundle=None (flag not given) + [output].bundle=false → no bundle.
    from meetscribe import pipeline

    rec = _bundle_run_env(tmp_path, monkeypatch)
    (tmp_path / "config.toml").write_text("[output]\nbundle = false\n")
    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out)) == 0
    assert not list(out.glob("*.mscribe"))


def test_run_bundle_flag_beats_config(tmp_path, monkeypatch):
    # an explicit True (the --bundle flag) wins over [output].bundle=false.
    from meetscribe import pipeline

    rec = _bundle_run_env(tmp_path, monkeypatch)
    (tmp_path / "config.toml").write_text("[output]\nbundle = false\n")
    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out), bundle=True) == 0
    assert (out / f"meeting-{out.name}.mscribe").exists()


def test_run_cleanup_none_resolves_from_config(tmp_path, monkeypatch):
    # cleanup=None (flag not given) + [output].cleanup=false → the built cleaner is
    # replaced by NullCleaner; without the config it runs (default on).
    import json

    from meetscribe import pipeline

    rec = tmp_path / "rec"
    (rec / "raw").mkdir(parents=True)
    _write_wav(rec / "raw" / "system.wav")
    monkeypatch.setenv("MEETSCRIBE_MODELS", _fake_models_dir(tmp_path))
    monkeypatch.setattr(
        pipeline, "build_components",
        lambda models_dir, num_speakers=-1, cleanup=True: Components(
            FakeVad(), FakeRecognizer(), FakeDiarizer(), FakeEmbedder(), UpperCleaner()
        ),
    )

    out1 = tmp_path / "out1"
    assert pipeline.run(audio=str(rec), out_dir=str(out1), bundle=False) == 0
    doc = json.loads((out1 / "transcript.json").read_text())
    assert doc["segments"][0]["text"] == "HELLO WORLD"  # default: cleanup on

    (tmp_path / "config.toml").write_text("[output]\ncleanup = false\n")
    out2 = tmp_path / "out2"
    assert pipeline.run(audio=str(rec), out_dir=str(out2), bundle=False) == 0
    doc = json.loads((out2 / "transcript.json").read_text())
    assert doc["segments"][0]["text"] == "hello world"  # config turned it off
    assert doc["cleaned"] is False


def test_run_malformed_config_exits_2_with_path_and_line(tmp_path, monkeypatch, capsys):
    # A typo'd config must abort loudly (exit 2, path + line) — never silently
    # degrade to defaults.
    from meetscribe import pipeline

    rec = _bundle_run_env(tmp_path, monkeypatch)
    (tmp_path / "config.toml").write_text("[output\nbundle = false\n")
    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out)) == 2
    printed = capsys.readouterr().out
    assert "config.toml" in printed and "line 1" in printed
    assert not (out / "transcript.json").exists()  # nothing was processed


def test_run_warns_on_unknown_config_key(tmp_path, monkeypatch):
    # Design: unknown sections/keys warn at LOAD time on every run (typo detection),
    # not only when the user happens to run doctor.
    from meetscribe import pipeline

    rec = _bundle_run_env(tmp_path, monkeypatch)
    (tmp_path / "config.toml").write_text("[output]\nbundel = false\n")  # typo'd key
    rep = RecordingReporter()
    assert pipeline.run(audio=str(rec), out_dir=str(tmp_path / "out"),
                        reporter=rep) == 0
    assert any("output.bundel" in w for w in rep.warns), rep.warns


def test_run_warns_on_insecure_api_key_perms(tmp_path, monkeypatch):
    # Design: [deepgram].api_key in a group/world-readable file warns once, at load.
    from meetscribe import pipeline

    rec = _bundle_run_env(tmp_path, monkeypatch)
    cfgp = tmp_path / "config.toml"
    cfgp.write_text('[deepgram]\napi_key = "dg_secret"\n')
    cfgp.chmod(0o644)
    rep = RecordingReporter()
    assert pipeline.run(audio=str(rec), out_dir=str(tmp_path / "out"),
                        reporter=rep) == 0
    assert any("0600" in w for w in rep.warns), rep.warns
    assert not any("dg_secret" in w for w in rep.warns)  # never echo the secret


def test_run_wrongly_typed_language_exits_2(tmp_path, monkeypatch, capsys):
    # Regression: [stt].language = 5 was resolved OUTSIDE the guarded block and
    # tracebacked (after the whole meeting, in the record flow). It must be a
    # clean exit 2 like every other typed-value error.
    from meetscribe import pipeline

    rec = _bundle_run_env(tmp_path, monkeypatch)
    (tmp_path / "config.toml").write_text(
        '[stt]\nbackend = "deepgram"\nlanguage = 5\n[deepgram]\napi_key = "k"\n'
    )
    assert pipeline.run(audio=str(rec), out_dir=str(tmp_path / "out")) == 2
    assert "language" in capsys.readouterr().out
    assert not (tmp_path / "out" / "transcript.json").exists()


def test_clean_existing_malformed_config_exits_2(tmp_path, monkeypatch, capsys):
    # Acceptance: EVERY subcommand exits 2 on a malformed config — clean included.
    from meetscribe import pipeline

    (tmp_path / "config.toml").write_text("[output\nbundle = false\n")
    src = tmp_path / "meeting"
    src.mkdir()
    assert pipeline.clean_existing(audio_dir=str(src)) == 2
    printed = capsys.readouterr().out
    assert "config.toml" in printed and "line 1" in printed


def test_clean_existing_writes_cleanup_dir_nondestructively(tmp_path, monkeypatch):
    from meetscribe import pipeline
    from meetscribe.output import write_meta, write_transcript
    from meetscribe.types import Utterance, Word

    src = tmp_path / "meeting"
    src.mkdir()
    utts = [Utterance(0.0, 1.0, "spk_0", "system", "hello world",
                      (Word("hello", 0.0, 0.5), Word("world", 0.5, 1.0)))]
    write_transcript(src / "transcript.json", "M", 1.0, utts)
    (src / "embeddings.npz").write_bytes(b"NPZ")
    write_meta(src / "meta.json", {"meeting_id": "M", "format_version": 2, "cleaned": False})
    original = (src / "transcript.json").read_text()

    monkeypatch.setenv("MEETSCRIBE_MODELS", _fake_models_dir(tmp_path))
    monkeypatch.setattr(
        pipeline, "build_components",
        lambda models_dir, num_speakers=-1, cleanup=True: Components(
            FakeVad(), FakeRecognizer(), FakeDiarizer(), FakeEmbedder(), UpperCleaner()
        ),
    )

    assert pipeline.clean_existing(audio_dir=str(src)) == 0

    out = tmp_path / "meeting-cleanup"
    assert out.is_dir()
    import json
    doc = json.loads((out / "transcript.json").read_text())
    assert doc["cleaned"] is True
    assert doc["segments"][0]["text"] == "HELLO WORLD"          # cleaned
    assert doc["segments"][0]["raw_text"] == "hello world"       # original preserved
    assert (out / "embeddings.npz").read_bytes() == b"NPZ"       # copied verbatim
    assert json.loads((out / "meta.json").read_text())["cleaned"] is True
    assert (src / "transcript.json").read_text() == original     # ORIGINAL untouched


def test_run_forwards_num_speakers_to_build_components(tmp_path, monkeypatch):
    # `process --speakers N` must reach the diarizer: run() hands num_speakers to
    # build_components (which passes it to OfflineDiarizer as num_clusters).
    from meetscribe import pipeline

    rec = tmp_path / "rec"
    (rec / "raw").mkdir(parents=True)
    _write_wav(rec / "raw" / "system.wav")

    monkeypatch.setenv("MEETSCRIBE_MODELS", _fake_models_dir(tmp_path))
    captured = {}

    def fake_build(models_dir, num_speakers=-1):
        captured["num_speakers"] = num_speakers
        return _components()

    monkeypatch.setattr(pipeline, "build_components", fake_build)

    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out), num_speakers=4) == 0
    assert captured["num_speakers"] == 4


# ---- backend resolution + remote (deepgram) wiring ----------------------------------

def test_resolve_backend_precedence(monkeypatch, tmp_path):
    # flag > STT_BACKEND env > [stt].backend config > default "local"
    from meetscribe.pipeline import resolve_backend

    monkeypatch.delenv("STT_BACKEND", raising=False)
    assert resolve_backend(None) == "local"
    (tmp_path / "config.toml").write_text('[stt]\nbackend = "deepgram"\n')
    assert resolve_backend(None) == "deepgram"  # config beats default
    monkeypatch.setenv("STT_BACKEND", "local")
    assert resolve_backend(None) == "local"  # env beats config
    monkeypatch.setenv("STT_BACKEND", "deepgram")
    assert resolve_backend("local") == "local"  # explicit flag beats env


def test_resolve_language_precedence(monkeypatch, tmp_path):
    # flag > STT_LANGUAGE env > [stt].language config > default "de"
    from meetscribe.pipeline import resolve_language

    monkeypatch.delenv("STT_LANGUAGE", raising=False)
    assert resolve_language(None) == "de"
    (tmp_path / "config.toml").write_text('[stt]\nlanguage = "multi"\n')
    assert resolve_language(None) == "multi"  # config beats default
    monkeypatch.setenv("STT_LANGUAGE", "en")
    assert resolve_language(None) == "en"  # env beats config
    assert resolve_language("multi") == "multi"  # explicit flag beats env


def test_check_backend_accepts_api_key_from_config(monkeypatch, tmp_path):
    # [deepgram].api_key beneath the env var satisfies the fail-fast check.
    from meetscribe.pipeline import check_backend

    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("STT_BACKEND", raising=False)
    (tmp_path / "config.toml").write_text('[deepgram]\napi_key = "dg_cfg_key"\n')
    name, err = check_backend("deepgram")
    assert name == "deepgram" and err is None


def test_check_backend_missing_key_mentions_env_and_config(monkeypatch, tmp_path):
    # The fail-fast message must offer BOTH fixes: the env var and the config key.
    from meetscribe.pipeline import check_backend

    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    name, err = check_backend("deepgram")
    assert err is not None
    assert "DEEPGRAM_API_KEY" in err
    assert "[deepgram].api_key" in err
    assert str(tmp_path / "config.toml") in err  # points at the config path in use


def _remote_rec_dir(tmp_path):
    rec = tmp_path / "rec"
    (rec / "raw").mkdir(parents=True)
    _write_wav(rec / "raw" / "mic.wav")
    _write_wav(rec / "raw" / "system.wav")
    return rec


def test_run_deepgram_without_key_exits_2(tmp_path, monkeypatch, capsys):
    # NO silent fallback to local: a missing key must abort with an actionable message.
    from meetscribe import pipeline

    rec = _remote_rec_dir(tmp_path)
    monkeypatch.setenv("MEETSCRIBE_MODELS", _fake_models_dir(tmp_path))
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)

    assert pipeline.run(audio=str(rec), out_dir=str(tmp_path / "out"),
                        backend="deepgram") == 2
    assert "DEEPGRAM_API_KEY" in capsys.readouterr().out
    assert not (tmp_path / "out" / "transcript.json").exists()  # nothing was processed


def test_run_unknown_backend_exits_2(tmp_path, monkeypatch, capsys):
    # A garbage STT_BACKEND must not be silently treated as local.
    from meetscribe import pipeline

    rec = _remote_rec_dir(tmp_path)
    monkeypatch.setenv("MEETSCRIBE_MODELS", _fake_models_dir(tmp_path))
    monkeypatch.setenv("STT_BACKEND", "whisper")

    assert pipeline.run(audio=str(rec), out_dir=str(tmp_path / "out")) == 2
    assert "whisper" in capsys.readouterr().out


def test_run_deepgram_skips_local_models_and_records_backend_meta(tmp_path, monkeypatch):
    """Remote mode must not load Parakeet/diarizer/VAD or the GGUF cleaner — only the
    local embedder (+ NullCleaner) — and meta.json must carry the remote identity."""
    import json

    from meetscribe import pipeline
    from meetscribe.backends import BackendResult
    from meetscribe.types import DiarSegment, Utterance, Word

    rec = _remote_rec_dir(tmp_path)
    monkeypatch.setenv("MEETSCRIBE_MODELS", _fake_models_dir(tmp_path))
    monkeypatch.setenv("DEEPGRAM_API_KEY", "k")
    monkeypatch.setenv("STT_BACKEND", "deepgram")  # env route (no flag) must work too

    def _boom(*a, **k):
        raise AssertionError("remote mode must not build the full local components")

    monkeypatch.setattr(pipeline, "build_components", _boom)
    monkeypatch.setattr(
        pipeline, "build_embed_components",
        lambda models_dir: Components(None, None, None, FakeEmbedder()),
    )

    seen = {}

    def fake_transcribe(self, mic, system, glossary, reporter):
        seen["language"] = self.config.language
        return BackendResult(
            mic_utts=[Utterance(0.0, 1.0, "me", "mic", "hi", (Word("hi", 0.0, 1.0),))],
            system_utts=[Utterance(0.0, 1.0, "spk_0", "system", "hallo",
                                   (Word("hallo", 0.0, 1.0),))],
            system_diar=[DiarSegment(0.0, 1.0, "spk_0")],
            models_meta={
                "asr_model": "deepgram-nova-3",
                "segmentation_model": "deepgram-diarizer",
                "backend_model_versions": {"name": "2-general-nova"},
                "request_ids": ["req-1", "req-2"],
            },
        )

    import meetscribe.deepgram as dg
    monkeypatch.setattr(dg.DeepgramBackend, "transcribe", fake_transcribe)

    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out), language="en") == 0
    assert seen["language"] == "en"  # --language reached the DeepgramConfig

    meta = json.loads((out / "meta.json").read_text())
    assert meta["backend"] == "deepgram"
    assert meta["asr_model"] == "deepgram-nova-3"
    assert meta["segmentation_model"] == "deepgram-diarizer"
    # remote substitute for local SHA pins travels with the artifact
    assert meta["backend_model_versions"] == {"name": "2-general-nova"}
    assert meta["request_ids"] == ["req-1", "req-2"]
    # embeddings stay local: identity + dim recorded exactly as in local mode
    assert meta["embedding_model"] == "3dspeaker_campplus_sv_zh_en_16k"
    assert meta["embedding_dim"] == 192
    assert meta["format_version"] == 2  # additions are additive; version unchanged

    # no cleanup in remote mode (NullCleaner): raw text, cleaned:false
    doc = json.loads((out / "transcript.json").read_text())
    assert doc["cleaned"] is False
    assert {s["speaker"] for s in doc["segments"]} == {"me", "spk_0"}


def _deepgram_run_env(tmp_path, monkeypatch):
    """Shared wiring for deepgram-mode run() tests: env + fake embed components +
    a scripted DeepgramBackend.transcribe."""
    from meetscribe import pipeline
    from meetscribe.backends import BackendResult
    from meetscribe.types import DiarSegment, Utterance, Word

    rec = _remote_rec_dir(tmp_path)
    monkeypatch.setenv("MEETSCRIBE_MODELS", _fake_models_dir(tmp_path))
    monkeypatch.setenv("DEEPGRAM_API_KEY", "k")
    monkeypatch.setattr(
        pipeline, "build_embed_components",
        lambda models_dir: Components(None, None, None, FakeEmbedder()),
    )

    def fake_transcribe(self, mic, system, glossary, reporter):
        return BackendResult(
            mic_utts=[Utterance(0.0, 1.0, "me", "mic", "hi", (Word("hi", 0.0, 1.0),))],
            system_utts=[Utterance(0.0, 1.0, "spk_0", "system", "hallo",
                                   (Word("hallo", 0.0, 1.0),))],
            system_diar=[DiarSegment(0.0, 1.0, "spk_0")],
            models_meta={"asr_model": "deepgram-nova-3",
                         "segmentation_model": "deepgram-diarizer"},
        )

    import meetscribe.deepgram as dg
    monkeypatch.setattr(dg.DeepgramBackend, "transcribe", fake_transcribe)
    return rec


def test_run_deepgram_warns_that_speakers_is_ignored(tmp_path, monkeypatch):
    # Deepgram's diarizer takes no forced cluster count: `--speakers 3` (or the
    # record-flow participant answer) must WARN, not be silently discarded.
    from meetscribe import pipeline

    rec = _deepgram_run_env(tmp_path, monkeypatch)
    rep = RecordingReporter()
    assert pipeline.run(audio=str(rec), out_dir=str(tmp_path / "out"),
                        backend="deepgram", num_speakers=3, reporter=rep) == 0
    assert any("speakers" in w for w in rep.warns), rep.warns


def test_run_deepgram_no_speakers_warning_when_automatic(tmp_path, monkeypatch):
    # No spurious warning when the count was never given (-1 = automatic).
    from meetscribe import pipeline

    rec = _deepgram_run_env(tmp_path, monkeypatch)
    rep = RecordingReporter()
    assert pipeline.run(audio=str(rec), out_dir=str(tmp_path / "out"),
                        backend="deepgram", num_speakers=-1, reporter=rep) == 0
    assert not any("speakers" in w for w in rep.warns), rep.warns


def test_run_deepgram_error_is_a_clean_exit_2(tmp_path, monkeypatch, capsys):
    # A present-but-invalid key (401 → DeepgramError) must surface as exit 2 with the
    # server's message — not an unhandled traceback after the meeting was recorded.
    from meetscribe import pipeline
    from meetscribe.deepgram import DeepgramError

    rec = _remote_rec_dir(tmp_path)
    monkeypatch.setenv("MEETSCRIBE_MODELS", _fake_models_dir(tmp_path))
    monkeypatch.setenv("DEEPGRAM_API_KEY", "invalid")
    monkeypatch.setattr(
        pipeline, "build_embed_components",
        lambda models_dir: Components(None, None, None, FakeEmbedder()),
    )

    def fake_transcribe(self, mic, system, glossary, reporter):
        raise DeepgramError("Deepgram rejected the request (HTTP 401) — check DEEPGRAM_API_KEY")

    import meetscribe.deepgram as dg
    monkeypatch.setattr(dg.DeepgramBackend, "transcribe", fake_transcribe)

    assert pipeline.run(audio=str(rec), out_dir=str(tmp_path / "out"),
                        backend="deepgram") == 2
    assert "DEEPGRAM_API_KEY" in capsys.readouterr().out


def test_run_deepgram_uses_config_api_key(tmp_path, monkeypatch):
    # With no DEEPGRAM_API_KEY in the env, the key from [deepgram].api_key must both
    # pass the preflight AND reach the DeepgramConfig actually used for the upload.
    from meetscribe import pipeline
    from meetscribe.backends import BackendResult
    from meetscribe.types import DiarSegment, Utterance, Word

    rec = _remote_rec_dir(tmp_path)
    monkeypatch.setenv("MEETSCRIBE_MODELS", _fake_models_dir(tmp_path))
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    (tmp_path / "config.toml").write_text('[deepgram]\napi_key = "dg_cfg_key"\n')
    monkeypatch.setattr(
        pipeline, "build_embed_components",
        lambda models_dir: Components(None, None, None, FakeEmbedder()),
    )

    seen = {}

    def fake_transcribe(self, mic, system, glossary, reporter):
        seen["api_key"] = self.config.api_key
        return BackendResult(
            mic_utts=[Utterance(0.0, 1.0, "me", "mic", "hi", (Word("hi", 0.0, 1.0),))],
            system_utts=[Utterance(0.0, 1.0, "spk_0", "system", "hallo",
                                   (Word("hallo", 0.0, 1.0),))],
            system_diar=[DiarSegment(0.0, 1.0, "spk_0")],
            models_meta={"asr_model": "deepgram-nova-3",
                         "segmentation_model": "deepgram-diarizer"},
        )

    import meetscribe.deepgram as dg
    monkeypatch.setattr(dg.DeepgramBackend, "transcribe", fake_transcribe)

    assert pipeline.run(audio=str(rec), out_dir=str(tmp_path / "out"),
                        backend="deepgram", bundle=False) == 0
    assert seen["api_key"] == "dg_cfg_key"


# ---- non-16 kHz input ---------------------------------------------------------------

class FixedBackend:
    """Backend returning a canned BackendResult (stands in for Deepgram: timestamps
    are true seconds regardless of the file's sample rate)."""

    name = "fixed"

    def __init__(self, result):
        self.result = result

    def transcribe(self, mic, system, glossary, reporter):
        return self.result


class SliceMeanEmbedder:
    dim = 192

    def __init__(self):
        self.means = []

    def embed(self, samples):
        self.means.append(float(np.mean(samples)))
        return np.ones(192, dtype=np.float32)


def test_process_resamples_non_16k_input(tmp_path):
    """A 48 kHz WAV (natural with the remote path: Deepgram reads the header, returns
    true-second timestamps) must not be sliced at 16 kHz offsets: that computed every
    turn/cluster vector from the wrong audio and inflated duration 3x — silently."""
    from meetscribe.backends import BackendResult
    from meetscribe.types import DiarSegment, Utterance, Word

    rate, seconds = 48000, 2.0
    n = int(rate * seconds)
    ramp = (np.linspace(0.0, 0.9, n) * 32767).astype("<i2")  # position-encoded content
    wav = tmp_path / "system.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(ramp.tobytes())

    embedder = SliceMeanEmbedder()
    comps = Components(None, None, None, embedder)
    backend = FixedBackend(BackendResult(
        mic_utts=[],
        system_utts=[Utterance(0.5, 1.5, "spk_0", "system", "hallo",
                               (Word("hallo", 0.5, 1.5),))],
        system_diar=[DiarSegment(0.5, 1.5, "spk_0")],
        models_meta={},
    ))
    rep = RecordingReporter()
    res = process(None, str(wav), comps, reporter=rep, backend=backend)

    assert res.duration_s == 2.0  # not 6.0 (96000 samples read as 16 kHz)
    # the 0.5–1.5 s slice sits at the middle of the ramp → mean ≈ 0.45; the unresampled
    # bug sliced the first sixth of the file instead (mean ≈ 0.15)
    assert embedder.means and all(0.4 < m < 0.5 for m in embedder.means), embedder.means
    assert any("48000" in w for w in rep.warns), rep.warns  # resampling is announced


def test_run_deepgram_failing_key_cmd_exits_2(tmp_path, monkeypatch, capsys):
    # The lazy api_key_cmd is fetched inside run()'s config guard: failure is a
    # clean exit 2 with the command's stderr, never a traceback.
    cfgfile = tmp_path / "config.toml"
    cfgfile.write_text('[deepgram]\napi_key_cmd = "echo kaboom >&2; exit 5"\n')
    monkeypatch.setenv("MEETSCRIBE_CONFIG", str(cfgfile))
    monkeypatch.setenv("MEETSCRIBE_MODELS", str(tmp_path))  # past the models gate
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("STT_BACKEND", raising=False)

    from meetscribe import pipeline
    assert pipeline.run(audio=str(tmp_path), backend="deepgram") == 2
    out = capsys.readouterr().out
    assert "api_key_cmd" in out and "kaboom" in out


def test_run_deepgram_key_cmd_supplies_key_material(tmp_path, monkeypatch):
    # The DeepgramConfig handed to the backend must contain the fetched material,
    # never the ApiKeyCmd marker.
    cfgfile = tmp_path / "config.toml"
    cfgfile.write_text('[deepgram]\napi_key_cmd = "echo dg_from_cmd"\n')
    monkeypatch.setenv("MEETSCRIBE_CONFIG", str(cfgfile))
    monkeypatch.setenv("MEETSCRIBE_MODELS", str(tmp_path))  # past the models gate
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("STT_BACKEND", raising=False)

    captured = {}
    import meetscribe.deepgram as dg

    class FakeBackend:
        def __init__(self, config, client=None):
            captured["api_key"] = config.api_key
            raise RuntimeError("stop here — key already captured")

    monkeypatch.setattr(dg, "DeepgramBackend", FakeBackend)

    meeting = tmp_path / "meeting"
    meeting.mkdir()
    (meeting / "system.wav").touch()

    from meetscribe import pipeline
    with pytest.raises(RuntimeError, match="stop here"):
        pipeline.run(audio=str(meeting), backend="deepgram")
    assert captured["api_key"] == "dg_from_cmd"


# ---- bk auto-upload wiring ----------------------------------------------------------
#
# The error-handling matrix (2026-09-09-phase2-bk-upload-design.md) is binding:
#   auto-upload on + missing base_url/token  → exit 2 in the config guard
#   auto-upload on + --no-bundle             → exit 2 naming both knobs
#   ANY upload-stage failure after processing → warn + retry hint, EXIT 0
#   success → bk-workflow.json + workflow_id/web_url printed


def _bk_run_env(tmp_path, monkeypatch, auto_upload=True):
    """A processable rec dir + a valid [bk] config (0600 — no perms-warning noise)."""
    rec = _bundle_run_env(tmp_path, monkeypatch)
    cfgp = tmp_path / "config.toml"
    cfgp.write_text(
        '[bk]\nbase_url = "https://bk.example.com"\ntoken = "tkn"\n'
        f"auto_upload = {'true' if auto_upload else 'false'}\n"
    )
    cfgp.chmod(0o600)
    return rec


def _fake_bk(monkeypatch, caps=None, caps_error=None, upload_error=None):
    """Patch the ONE seam — the bk.BkClient module symbol — with a scriptable fake.

    Returns a state dict; ``state["clients"]`` collects every constructed client so
    tests can assert config, calls, and (crucially) *non*-construction."""
    import meetscribe.bk as bk_mod

    state: dict = {"clients": []}

    class FakeBkClient:
        def __init__(self, config, opener=None, sleep=None):
            self.config = config
            self.uploads: list = []
            state["clients"].append(self)

        def capabilities(self):
            if caps_error is not None:
                raise bk_mod.BkError(caps_error)
            if caps is not None:
                return caps
            return {
                "contract_version": 1,
                "mscribe_format_versions": [2],
                "max_bundle_bytes": 52428800,
            }

        def upload_bundle(self, bundle_path, *, meeting_id, calendar_meeting_id=None):
            if upload_error is not None:
                raise bk_mod.BkError(upload_error)
            self.uploads.append((str(bundle_path), meeting_id, calendar_meeting_id))
            return {
                "workflow_id": "wf_01j9",
                "state_url": "/api/meetscribe/v1/workflows/wf_01j9",
                "web_url": "https://bk.example.com/workflows/wf_01j9",
            }

    monkeypatch.setattr(bk_mod, "BkClient", FakeBkClient)
    return state


def test_run_upload_success_writes_ref_and_prints_workflow(tmp_path, monkeypatch, capsys):
    from meetscribe import bk as bk_mod
    from meetscribe import pipeline

    rec = _bk_run_env(tmp_path, monkeypatch)
    state = _fake_bk(monkeypatch)
    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out)) == 0

    (client,) = state["clients"]
    assert client.config.base_url == "https://bk.example.com"
    assert client.config.token == "tkn"
    [(bundle_path, meeting_id, cal)] = client.uploads
    assert bundle_path.endswith(f"meeting-{out.name}.mscribe")
    assert meeting_id == out.name  # Idempotency-Key = the meeting id
    assert cal is None  # ?meeting_id= is Phase 3

    ref = bk_mod.read_workflow_ref(out)
    assert ref is not None
    assert ref["workflow_id"] == "wf_01j9"
    assert ref["web_url"] == "https://bk.example.com/workflows/wf_01j9"
    assert ref["state_url"] == "/api/meetscribe/v1/workflows/wf_01j9"
    assert ref["state"] == "processing"
    assert ref["uploaded_at"] and ref["checked_at"]

    printed = capsys.readouterr().out
    assert "wf_01j9" in printed
    assert "https://bk.example.com/workflows/wf_01j9" in printed


def test_run_upload_off_by_default_never_constructs_client(tmp_path, monkeypatch):
    # No [bk] config, upload=None → byte-identical behavior: no client, no ref.
    import meetscribe.bk as bk_mod
    from meetscribe import pipeline

    rec = _bundle_run_env(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise AssertionError("BkClient must not be constructed when upload is off")

    monkeypatch.setattr(bk_mod, "BkClient", boom)
    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out)) == 0
    assert bk_mod.read_workflow_ref(out) is None


def test_run_upload_flag_false_beats_config(tmp_path, monkeypatch):
    # --no-upload (upload=False) wins over [bk].auto_upload = true.
    import meetscribe.bk as bk_mod
    from meetscribe import pipeline

    rec = _bk_run_env(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise AssertionError("--no-upload must win over [bk].auto_upload")

    monkeypatch.setattr(bk_mod, "BkClient", boom)
    assert pipeline.run(audio=str(rec), out_dir=str(tmp_path / "out"),
                        upload=False) == 0


def test_run_upload_flag_true_beats_config_off(tmp_path, monkeypatch):
    # --upload (upload=True) wins over [bk].auto_upload = false.
    from meetscribe import pipeline

    rec = _bk_run_env(tmp_path, monkeypatch, auto_upload=False)
    state = _fake_bk(monkeypatch)
    assert pipeline.run(audio=str(rec), out_dir=str(tmp_path / "out"),
                        upload=True) == 0
    assert state["clients"] and state["clients"][0].uploads


def test_run_upload_missing_bk_config_exits_2(tmp_path, monkeypatch, capsys):
    # Matrix row 1: auto-upload on, base_url/token missing → exit 2, fail fast.
    from meetscribe import pipeline

    rec = _bundle_run_env(tmp_path, monkeypatch)
    (tmp_path / "config.toml").write_text("[bk]\nauto_upload = true\n")
    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out)) == 2
    printed = capsys.readouterr().out
    assert "[bk].base_url" in printed and "MEETSCRIBE_BK_TOKEN" in printed
    assert not (out / "transcript.json").exists()  # nothing was processed


def test_run_upload_failing_token_cmd_exits_2(tmp_path, monkeypatch, capsys):
    # The lazy token_cmd is fetched EAGERLY inside run()'s config guard (mirror of
    # the deepgram api_key_cmd): failure is a clean exit 2, never a traceback.
    from meetscribe import pipeline

    rec = _bundle_run_env(tmp_path, monkeypatch)
    (tmp_path / "config.toml").write_text(
        '[bk]\nauto_upload = true\nbase_url = "https://bk.example.com"\n'
        'token_cmd = "echo kaboom >&2; exit 5"\n'
    )
    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out)) == 2
    printed = capsys.readouterr().out
    assert "token_cmd" in printed and "kaboom" in printed
    assert not (out / "transcript.json").exists()  # nothing was processed


def test_run_upload_token_cmd_supplies_material(tmp_path, monkeypatch):
    # The BkConfig handed to the client must carry the fetched material, never the
    # SecretCmd marker.
    from meetscribe import pipeline

    rec = _bundle_run_env(tmp_path, monkeypatch)
    (tmp_path / "config.toml").write_text(
        '[bk]\nauto_upload = true\nbase_url = "https://bk.example.com"\n'
        'token_cmd = "echo tok_from_cmd"\n'
    )
    state = _fake_bk(monkeypatch)
    assert pipeline.run(audio=str(rec), out_dir=str(tmp_path / "out")) == 0
    assert state["clients"][0].config.token == "tok_from_cmd"


def test_run_upload_with_no_bundle_exits_2(tmp_path, monkeypatch, capsys):
    # Matrix row 2: auto-upload needs the bundle — the conflict exits 2 BEFORE any
    # processing, and the message names both knobs.
    from meetscribe import pipeline

    rec = _bk_run_env(tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out), bundle=False) == 2
    printed = capsys.readouterr().out
    assert "--no-bundle" in printed and "[bk].auto_upload" in printed
    assert not (out / "transcript.json").exists()  # nothing was processed


def test_run_upload_network_failure_warns_and_exits_0(tmp_path, monkeypatch, capsys):
    # Matrix: upload network/5xx after retries → loud warn + retry hint, EXIT 0 —
    # the artifacts are complete and local; the retry path is first-class.
    from meetscribe import bk as bk_mod
    from meetscribe import pipeline

    rec = _bk_run_env(tmp_path, monkeypatch)
    _fake_bk(monkeypatch,
             upload_error="bk request failed after 3 attempts (network error: down)")
    rep = RecordingReporter()
    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out), reporter=rep) == 0
    assert any("upload" in w for w in rep.warns), rep.warns
    assert f"meetscribe upload {out}" in capsys.readouterr().out  # the retry hint
    # artifacts + bundle intact; no half-written workflow ref
    assert (out / "transcript.json").exists()
    assert (out / f"meeting-{out.name}.mscribe").exists()
    assert bk_mod.read_workflow_ref(out) is None


def test_run_upload_capabilities_failure_warns_and_exits_0(tmp_path, monkeypatch, capsys):
    # A failing capabilities call (e.g. 404 — no bk API there) degrades identically.
    from meetscribe import bk as bk_mod
    from meetscribe import pipeline

    rec = _bk_run_env(tmp_path, monkeypatch)
    _fake_bk(monkeypatch, caps_error="bk request failed (HTTP 404: not found)")
    rep = RecordingReporter()
    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out), reporter=rep) == 0
    assert any("404" in w for w in rep.warns), rep.warns
    assert f"meetscribe upload {out}" in capsys.readouterr().out
    assert bk_mod.read_workflow_ref(out) is None


def test_run_upload_preflight_failure_skips_upload_and_exits_0(tmp_path, monkeypatch, capsys):
    # Matrix: preflight failure (unsupported format_version) → upload SKIPPED,
    # warn + retry hint, exit 0.
    from meetscribe import bk as bk_mod
    from meetscribe import pipeline

    rec = _bk_run_env(tmp_path, monkeypatch)
    state = _fake_bk(monkeypatch,
                     caps={"contract_version": 1, "mscribe_format_versions": [1]})
    rep = RecordingReporter()
    out = tmp_path / "out"
    assert pipeline.run(audio=str(rec), out_dir=str(out), reporter=rep) == 0
    assert state["clients"][0].uploads == []  # never POSTed
    assert any("format_version" in w for w in rep.warns), rep.warns
    assert f"meetscribe upload {out}" in capsys.readouterr().out
    assert bk_mod.read_workflow_ref(out) is None
