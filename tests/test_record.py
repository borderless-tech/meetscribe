"""record.py — device parsing, name matching, ffmpeg command building (both OS).

Pure helpers only; no ffmpeg is executed here (the real Linux capture is a separate, skippable
integration test). Device indices are matched by NAME — never hard-coded (§5).
"""

from pathlib import Path

import numpy as np
import pytest

from meetscribe.record import (
    _drive_meters,
    _popen_kwargs,
    build_linux_ffmpeg_cmd,
    build_macos_ffmpeg_cmd,
    find_monitor_source,
    match_device,
    parse_astats,
    parse_avfoundation_devices,
    parse_pw_default_sink,
    parse_pw_sources,
    rms,
    stop_ffmpeg,
    warn_if_silent,
)

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    """No test may read the real home: the config is a per-test tmp file (missing by
    default → all defaults) and the XDG dirs point into tmp_path (the record flow also
    touches the glossary under XDG_CONFIG_HOME). Tests that need a config layer write
    ``tmp_path / "config.toml"``. The STT env vars are scrubbed too — a developer
    shell exporting STT_BACKEND/DEEPGRAM_API_KEY must not leak into run() tests."""
    monkeypatch.setenv("MEETSCRIBE_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    for var in ("STT_BACKEND", "STT_LANGUAGE", "DEEPGRAM_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# ---- macOS device parsing / matching -------------------------------------------------

def test_parse_avfoundation_audio_devices_only():
    text = (FIX / "avfoundation_devices.txt").read_text()
    devices = parse_avfoundation_devices(text)
    assert devices == [(0, "Built-in Microphone"), (1, "meetscribe"), (2, "BlackHole 2ch")]


def test_match_device_by_name_returns_index():
    devices = [(0, "Built-in Microphone"), (1, "meetscribe"), (2, "BlackHole 2ch")]
    assert match_device(devices, "meetscribe") == 1


def test_match_device_missing_raises():
    with pytest.raises(ValueError, match="nope"):
        match_device([(0, "Built-in Microphone")], "nope")


def test_macos_cmd_has_pan_split_and_two_outputs():
    cmd = build_macos_ffmpeg_cmd(1, "raw/mic.wav", "raw/system.wav")
    joined = " ".join(cmd)
    assert "avfoundation" in joined
    assert '-i' in cmd and ':1' in joined            # aggregate device index, matched by name
    assert "pan=" in joined                          # channel split, not stereo downmix
    assert cmd.count("-map") == 2
    assert "raw/mic.wav" in cmd and "raw/system.wav" in cmd


# ---- Linux source parsing / command --------------------------------------------------

def test_parse_pw_sources_returns_names():
    text = (FIX / "pactl_sources.txt").read_text()
    names = parse_pw_sources(text)
    assert "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor" in names
    assert "alsa_input.pci-0000_00_1f.3.analog-stereo" in names


def test_find_monitor_source_prefers_monitor():
    names = parse_pw_sources((FIX / "pactl_sources.txt").read_text())
    assert find_monitor_source(names).endswith(".monitor")


def test_find_monitor_source_prefers_default_sink_monitor():
    # Regression: the active output (a Bluetooth headset) sorts LAST among the enumerated
    # monitors, while a silent unused HDMI output sorts first. meetscribe must tap the DEFAULT
    # sink's monitor — not blindly the first monitor — or the system track records silence.
    names = [
        "alsa_output.hdmi3.monitor",     # first, but nothing plays here → silent
        "alsa_output.hdmi2.monitor",
        "alsa_input.builtin_mic",
        "bluez_output.AC_80_0A_F3_FB_F1.1.monitor",  # the real output, sorts last
    ]
    got = find_monitor_source(names, default_sink="bluez_output.AC_80_0A_F3_FB_F1.1")
    assert got == "bluez_output.AC_80_0A_F3_FB_F1.1.monitor"


def test_find_monitor_source_without_default_returns_first_monitor():
    # No default known → preserve the old best-effort behavior (first monitor wins).
    names = ["alsa_output.hdmi3.monitor", "bluez_output.headset.monitor"]
    assert find_monitor_source(names) == "alsa_output.hdmi3.monitor"


def test_find_monitor_source_default_without_monitor_falls_back_to_first():
    # Default sink is known but exposes no matching monitor → fall back to first monitor.
    names = ["alsa_output.hdmi3.monitor", "alsa_output.headphones.monitor"]
    assert find_monitor_source(names, default_sink="sink_without_monitor") \
        == "alsa_output.hdmi3.monitor"


def test_parse_pw_default_sink_reads_metadata():
    text = (FIX / "pw_dump_default.json").read_text()
    assert parse_pw_default_sink(text) == "bluez_output.AC_80_0A_F3_FB_F1.1"


def test_parse_pw_default_sink_missing_returns_none():
    assert parse_pw_default_sink("[]") is None


def test_linux_cmd_has_two_pulse_inputs_and_two_outputs():
    cmd = build_linux_ffmpeg_cmd("mic_src", "sink.monitor", "raw/mic.wav", "raw/system.wav")
    assert cmd.count("pulse") == 2
    assert "mic_src" in cmd and "sink.monitor" in cmd
    assert cmd.count("-map") == 2
    assert cmd[-1] == "raw/system.wav"


# ---- RMS / silence / graceful stop ---------------------------------------------------

def test_rms_of_known_buffer():
    assert rms(np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)) == pytest.approx(1.0)


def test_rms_zero_for_silence():
    assert rms(np.zeros(100, dtype=np.float32)) == 0.0


def test_warn_if_silent_flags_silent_wav(tmp_path):
    import wave

    p = tmp_path / "silent.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    msg = warn_if_silent(p)
    assert msg is not None and "silent" in msg.lower()


## ---- astats metering ---------------------------------------------------------------

def test_parse_astats_per_channel():
    assert parse_astats("lavfi.astats.1.RMS_level=-29.537922") == (1, -29.537922)
    assert parse_astats("lavfi.astats.2.RMS_level=-16.258925") == (2, -16.258925)


def test_parse_astats_inf_is_negative_infinity():
    assert parse_astats("lavfi.astats.1.RMS_level=-inf") == (1, float("-inf"))


def test_parse_astats_non_matching_returns_none():
    assert parse_astats("frame:0    pts:0       pts_time:0") is None
    assert parse_astats("lavfi.astats.Overall.Peak_level=-3.0") is None


def test_linux_metered_cmd_has_astats_amerge_and_two_wavs():
    cmd = build_linux_ffmpeg_cmd("mic_src", "sink.monitor", "mic.wav", "sys.wav", metered=True)
    j = " ".join(cmd)
    assert "astats" in j and "amerge" in j and "ametadata" in j
    assert j.count("pulse") == 2
    assert "mic.wav" in cmd and "sys.wav" in cmd


def test_macos_metered_cmd_has_astats_and_pan():
    cmd = build_macos_ffmpeg_cmd(1, "mic.wav", "sys.wav", metered=True)
    j = " ".join(cmd)
    assert "astats" in j and "amerge" in j and "pan=" in j


def test_record_cmds_suppress_ffmpeg_log_noise():
    # ffmpeg's banner + astats per-frame report go to stderr, where the rich UI also renders;
    # -loglevel error -nostats keeps them from interleaving (meter data is on stdout, unaffected).
    for cmd in (
        build_linux_ffmpeg_cmd("m", "s", "a", "b"),
        build_linux_ffmpeg_cmd("m", "s", "a", "b", metered=True),
        build_macos_ffmpeg_cmd(0, "a", "b"),
        build_macos_ffmpeg_cmd(0, "a", "b", metered=True),
    ):
        assert "-loglevel" in cmd and "error" in cmd
        assert "-nostats" in cmd


def test_non_metered_cmds_unchanged():
    # default (metered=False) keeps the simple two-output command
    assert build_linux_ffmpeg_cmd("m", "s", "a", "b").count("-map") == 2
    assert "astats" not in " ".join(build_macos_ffmpeg_cmd(0, "a", "b"))


def test_popen_kwargs_metered_keeps_stdin_binary():
    import subprocess

    # metered must NOT use text mode: stop_ffmpeg writes b"q" to stdin, and a text-mode
    # stdin would raise TypeError inside the SIGINT handler (regression guard).
    kw = _popen_kwargs(metered=True)
    assert kw.get("text") in (None, False)
    assert kw["stdin"] == subprocess.PIPE
    assert kw["stdout"] == subprocess.PIPE
    # non-metered: no stdout pipe needed
    assert "stdout" not in _popen_kwargs(metered=False)


def test_drive_meters_parses_bytes_and_updates_channels():
    class RecMeters:
        def __init__(self):
            self.updates = []

        def update(self, ch, db):
            self.updates.append((ch, db))

    stream = [
        b"frame:0    pts_time:0\n",
        b"lavfi.astats.1.RMS_level=-20.0\n",
        b"lavfi.astats.2.RMS_level=-10.0\n",
        b"lavfi.astats.Overall.Peak_level=-3.0\n",  # ignored (not per-channel RMS)
    ]
    m = RecMeters()
    _drive_meters(iter(stream), m)
    assert m.updates == [(1, -20.0), (2, -10.0)]


def test_stop_ffmpeg_tolerates_write_typeerror():
    # defense-in-depth: a mode mismatch must not propagate out of the SIGINT handler.
    class BadStdin:
        def write(self, b):
            raise TypeError("write() argument must be str, not bytes")

        def flush(self):
            pass

    class Proc:
        stdin = BadStdin()

        def wait(self, timeout=None):
            return 0

    stop_ffmpeg(Proc())  # must not raise


def test_stop_ffmpeg_sends_q_not_kill():
    class FakeStdin:
        def __init__(self):
            self.written = b""
            self.flushed = False

        def write(self, b):
            self.written += b

        def flush(self):
            self.flushed = True

    class FakeProc:
        def __init__(self):
            self.stdin = FakeStdin()
            self.killed = False
            self.waited = False

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.waited = True
            return 0

    proc = FakeProc()
    stop_ffmpeg(proc)
    assert proc.stdin.written == b"q"
    assert proc.stdin.flushed
    assert not proc.killed


# ---- run() threads the real capture start into the pipeline --------------------------

def test_record_tracks_override_bypasses_enumeration(tmp_path, monkeypatch):
    # --system-source is an explicit escape hatch: it must be used verbatim as the system
    # input, without consulting (or needing) the sink/monitor enumeration at all.
    import subprocess

    from meetscribe import record

    monkeypatch.setattr(record.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        record, "_list_linux_sources",
        lambda: (_ for _ in ()).throw(AssertionError("enumeration must be skipped")),
    )

    captured = {}

    class FakeProc:
        stdin = None
        stdout = None

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    record.record_tracks(
        str(tmp_path / "mic.wav"), str(tmp_path / "sys.wav"),
        duration=1.0, system_source="my_sink.monitor",
    )
    assert "my_sink.monitor" in captured["cmd"]


def test_run_forwards_system_source_override(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "meetscribe.record.record_tracks",
        lambda *a, **k: captured.update(k),
    )
    monkeypatch.setattr("meetscribe.record.warn_if_silent", lambda p: None)
    import meetscribe.pipeline as pl
    monkeypatch.setattr(pl, "run", lambda **k: 0)

    from meetscribe import record
    record.run(out_dir=str(tmp_path / "m"), system_source="my_sink.monitor")
    assert captured["system_source"] == "my_sink.monitor"


def test_run_passes_started_at_and_bundle_to_pipeline(tmp_path, monkeypatch):
    # record.run knows the real wall-clock meeting start; it must hand a tz-aware
    # started_at (and the --bundle flag) to pipeline.run rather than let the pipeline
    # fall back to the WAV mtime.
    captured = {}
    monkeypatch.setattr("meetscribe.record.record_tracks", lambda *a, **k: None)
    monkeypatch.setattr("meetscribe.record.warn_if_silent", lambda p: None)
    import meetscribe.pipeline as pl
    monkeypatch.setattr(pl, "run", lambda **k: captured.update(k) or 0)

    from meetscribe import record
    assert record.run(out_dir=str(tmp_path / "m"), bundle=True) == 0

    assert captured["bundle"] is True
    assert captured["started_at"] is not None  # a tz-aware datetime
    assert captured["started_at"].tzinfo is not None


# ---- post-recording participant prompt -----------------------------------------------

def test_participants_to_speakers_subtracts_the_user():
    from meetscribe.record import participants_to_speakers

    assert participants_to_speakers("5") == 4
    assert participants_to_speakers(" 2 ") == 1


def test_participants_to_speakers_blank_or_invalid_means_automatic():
    from meetscribe.record import participants_to_speakers

    assert participants_to_speakers("") == -1
    assert participants_to_speakers("abc") == -1
    # 1 participant = only the user → nothing to force on the system track
    assert participants_to_speakers("1") == -1
    assert participants_to_speakers("0") == -1
    assert participants_to_speakers("-3") == -1


def test_ask_participants_maps_answer():
    from meetscribe.record import ask_participants

    assert ask_participants(input_fn=lambda prompt: "4") == 3


def test_ask_participants_eof_or_interrupt_skips():
    from meetscribe.record import ask_participants

    def eof(prompt):
        raise EOFError

    def interrupt(prompt):
        raise KeyboardInterrupt

    assert ask_participants(input_fn=eof) == -1
    assert ask_participants(input_fn=interrupt) == -1


def test_run_forwards_participant_answer_to_pipeline(tmp_path, monkeypatch):
    # After recording stops, run() asks for the participant count and hands the
    # derived speaker count to pipeline.run.
    captured = {}
    monkeypatch.setattr("meetscribe.record.record_tracks", lambda *a, **k: None)
    monkeypatch.setattr("meetscribe.record.warn_if_silent", lambda p: None)
    monkeypatch.setattr("meetscribe.record.ask_participants", lambda: 3)
    import meetscribe.pipeline as pl
    monkeypatch.setattr(pl, "run", lambda **k: captured.update(k) or 0)

    from meetscribe import record
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO())  # not a tty
    monkeypatch.setattr("meetscribe.record._stdin_is_tty", lambda: True)
    assert record.run(out_dir=str(tmp_path / "m")) == 0
    assert captured["num_speakers"] == 3


def test_ask_names_collects_until_blank():
    from meetscribe.record import ask_participant_names

    answers = iter(["Georg", "Christian", ""])
    assert ask_participant_names(4, input_fn=lambda _: next(answers)) == ["Georg", "Christian"]


def test_ask_names_skipped_when_count_below_two():
    from meetscribe.record import ask_participant_names

    assert ask_participant_names(1, input_fn=lambda _: "x") == []


def test_ask_names_stops_at_count():
    from meetscribe.record import ask_participant_names

    # count=2 → at most 2 prompts even if the user keeps typing
    assert ask_participant_names(2, input_fn=lambda _: "X") == ["X", "X"]


def test_ask_names_eof_keeps_prior():
    from meetscribe.record import ask_participant_names

    calls = {"n": 0}

    def boom(_):
        if calls["n"] == 0:
            calls["n"] += 1
            return "Georg"
        raise EOFError

    assert ask_participant_names(3, input_fn=boom) == ["Georg"]


def test_run_skips_prompt_without_tty(tmp_path, monkeypatch):
    # Non-interactive stdin (scripts, CI): never block on input(); use automatic mode.
    captured = {}
    monkeypatch.setattr("meetscribe.record.record_tracks", lambda *a, **k: None)
    monkeypatch.setattr("meetscribe.record.warn_if_silent", lambda p: None)
    monkeypatch.setattr("meetscribe.record._stdin_is_tty", lambda: False)
    import meetscribe.pipeline as pl
    monkeypatch.setattr(pl, "run", lambda **k: captured.update(k) or 0)

    from meetscribe import record
    assert record.run(out_dir=str(tmp_path / "m")) == 0
    assert captured["num_speakers"] == -1


def test_run_forwards_backend_and_language(tmp_path, monkeypatch):
    # record.run must hand --backend/--language through to pipeline.run untouched
    # (resolution — flag > env > default — happens once, in the pipeline).
    captured = {}
    monkeypatch.setattr("meetscribe.record.record_tracks", lambda *a, **k: None)
    monkeypatch.setattr("meetscribe.record.warn_if_silent", lambda p: None)
    monkeypatch.setattr("meetscribe.record._stdin_is_tty", lambda: False)
    monkeypatch.delenv("STT_BACKEND", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "k")  # record.run preflights the key
    import meetscribe.pipeline as pl
    monkeypatch.setattr(pl, "run", lambda **k: captured.update(k) or 0)

    from meetscribe import record
    assert record.run(out_dir=str(tmp_path / "m"), backend="deepgram", language="en") == 0
    assert captured["backend"] == "deepgram"
    assert captured["language"] == "en"

    assert record.run(out_dir=str(tmp_path / "m")) == 0
    assert captured["backend"] is None  # default: let the pipeline resolve env
    assert captured["language"] is None


def test_run_deepgram_without_key_fails_before_recording(tmp_path, monkeypatch, capsys):
    # Non-negotiable: a missing DEEPGRAM_API_KEY must fail BEFORE ffmpeg starts —
    # not after the user recorded an hour-long meeting and answered the prompts.
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("STT_BACKEND", raising=False)

    def boom(*a, **k):
        raise AssertionError("record_tracks must not run without a usable backend")

    monkeypatch.setattr("meetscribe.record.record_tracks", boom)

    from meetscribe import record
    assert record.run(out_dir=str(tmp_path / "m"), backend="deepgram") == 2
    assert "DEEPGRAM_API_KEY" in capsys.readouterr().out
    assert not (tmp_path / "m").exists()  # nothing was created either


def test_run_unknown_backend_env_fails_before_recording(tmp_path, monkeypatch, capsys):
    # A typo'd STT_BACKEND env value bypasses argparse choices — it must still fail
    # before recording starts, not an hour later in pipeline.run.
    monkeypatch.setenv("STT_BACKEND", "deepgramm")

    def boom(*a, **k):
        raise AssertionError("record_tracks must not run with an unknown backend")

    monkeypatch.setattr("meetscribe.record.record_tracks", boom)

    from meetscribe import record
    assert record.run(out_dir=str(tmp_path / "m")) == 2
    assert "deepgramm" in capsys.readouterr().out


# ---- config-backed resolution (meetings_dir / system_source / bundle / cleanup) -------

def _stub_run(monkeypatch, captured):
    """Stub the capture + pipeline hand-off; ``captured`` collects pipeline.run kwargs."""
    monkeypatch.setattr("meetscribe.record.record_tracks", lambda *a, **k: None)
    monkeypatch.setattr("meetscribe.record.warn_if_silent", lambda p: None)
    monkeypatch.setattr("meetscribe.record._stdin_is_tty", lambda: False)
    import meetscribe.pipeline as pl
    monkeypatch.setattr(pl, "run", lambda **k: captured.update(k) or 0)


def test_run_default_out_dir_under_meetings_dir(tmp_path, monkeypatch):
    # Without -o, recordings land under <meetings_dir>/meetscribe-<timestamp>/ —
    # $XDG_DATA_HOME/meetscribe/meetings by default — instead of littering the CWD.
    captured = {}
    _stub_run(monkeypatch, captured)

    from meetscribe import record
    assert record.run() == 0
    root = Path(captured["audio"])
    assert root.parent == tmp_path / "xdg-data" / "meetscribe" / "meetings"
    assert root.name.startswith("meetscribe-")
    assert (root / "raw").is_dir()  # created with parents


def test_run_meetings_dir_from_config(tmp_path, monkeypatch):
    # [storage].meetings_dir beats the XDG default; -o (out_dir) still wins per run.
    captured = {}
    _stub_run(monkeypatch, captured)
    (tmp_path / "config.toml").write_text(
        f'[storage]\nmeetings_dir = "{tmp_path / "mtg"}"\n'
    )

    from meetscribe import record
    assert record.run() == 0
    assert Path(captured["audio"]).parent == tmp_path / "mtg"

    assert record.run(out_dir=str(tmp_path / "explicit")) == 0
    assert Path(captured["audio"]) == tmp_path / "explicit"  # -o untouched


def test_run_system_source_from_config(tmp_path, monkeypatch):
    # [record].system_source fills in when the flag is absent; the flag wins.
    captured_pl: dict = {}
    captured_rec: dict = {}
    _stub_run(monkeypatch, captured_pl)
    monkeypatch.setattr(
        "meetscribe.record.record_tracks", lambda *a, **k: captured_rec.update(k)
    )
    (tmp_path / "config.toml").write_text('[record]\nsystem_source = "cfg_sink.monitor"\n')

    from meetscribe import record
    assert record.run(out_dir=str(tmp_path / "m")) == 0
    assert captured_rec["system_source"] == "cfg_sink.monitor"

    assert record.run(out_dir=str(tmp_path / "m"), system_source="flag.monitor") == 0
    assert captured_rec["system_source"] == "flag.monitor"


def test_run_resolves_bundle_and_cleanup(tmp_path, monkeypatch):
    # None = "resolve from config" (contract with the CLI): default on, config layer
    # beneath, an explicit flag on top.
    captured = {}
    _stub_run(monkeypatch, captured)

    from meetscribe import record
    assert record.run(out_dir=str(tmp_path / "m")) == 0
    assert captured["bundle"] is True and captured["cleanup"] is True  # defaults on

    (tmp_path / "config.toml").write_text("[output]\nbundle = false\ncleanup = false\n")
    assert record.run(out_dir=str(tmp_path / "m")) == 0
    assert captured["bundle"] is False and captured["cleanup"] is False  # config layer

    assert record.run(out_dir=str(tmp_path / "m"), bundle=True, cleanup=True) == 0
    assert captured["bundle"] is True and captured["cleanup"] is True  # flag wins


def test_run_malformed_config_fails_before_recording(tmp_path, monkeypatch, capsys):
    # A typo'd config must abort (exit 2, path + line) BEFORE ffmpeg starts.
    (tmp_path / "config.toml").write_text("[output\nbundle = false\n")

    def boom(*a, **k):
        raise AssertionError("record_tracks must not run with a malformed config")

    monkeypatch.setattr("meetscribe.record.record_tracks", boom)

    from meetscribe import record
    assert record.run(out_dir=str(tmp_path / "m")) == 2
    printed = capsys.readouterr().out
    assert "config.toml" in printed and "line 1" in printed
    assert not (tmp_path / "m").exists()  # nothing was created either


class _WarnReporter:
    """Minimal reporter capturing warn() calls (record.run only warns directly)."""

    def __init__(self):
        self.warns = []

    def warn(self, msg):
        self.warns.append(msg)

    def info(self, msg):
        pass


def test_run_warns_on_unknown_config_key(tmp_path, monkeypatch):
    # Design: a typo'd key warns at LOAD time on every record run — the user must
    # not need to run doctor to learn their config is being ignored.
    captured = {}
    _stub_run(monkeypatch, captured)
    (tmp_path / "config.toml").write_text("[output]\nbundel = false\n")

    from meetscribe import record
    rep = _WarnReporter()
    assert record.run(out_dir=str(tmp_path / "m"), reporter=rep) == 0
    assert any("output.bundel" in w for w in rep.warns), rep.warns


def test_run_wrongly_typed_language_fails_before_recording(tmp_path, monkeypatch, capsys):
    # A wrongly-typed [stt].language previously crashed only in the pipeline
    # hand-off — AFTER the whole meeting was recorded. Preflight must catch it.
    (tmp_path / "config.toml").write_text("[stt]\nlanguage = 5\n")

    def boom(*a, **k):
        raise AssertionError("record_tracks must not run with a broken config")

    monkeypatch.setattr("meetscribe.record.record_tracks", boom)

    from meetscribe import record
    assert record.run(out_dir=str(tmp_path / "m")) == 2
    assert "language" in capsys.readouterr().out
    assert not (tmp_path / "m").exists()  # nothing was created either


def test_run_deepgram_key_from_config_passes_preflight(tmp_path, monkeypatch):
    # [deepgram].api_key (no env var) satisfies the fail-fast preflight.
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("STT_BACKEND", raising=False)
    (tmp_path / "config.toml").write_text('[deepgram]\napi_key = "dg_cfg_key"\n')
    captured = {}
    _stub_run(monkeypatch, captured)

    from meetscribe import record
    assert record.run(out_dir=str(tmp_path / "m"), backend="deepgram") == 0
    assert captured["backend"] == "deepgram"
