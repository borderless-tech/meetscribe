"""Reporter seam: NullReporter no-ops, RichReporter smoke + non-TTY plain fallback."""

import io
import re

from meetscribe.progress import (
    NullReporter,
    RichReporter,
    Summary,
    _fmt_duration,
    rms_to_meter,
)

ANSI = re.compile(r"\x1b\[")


def test_null_reporter_stage_is_a_noop_context_manager(capsys):
    r = NullReporter()
    with r.stage("resample"):
        pass
    assert capsys.readouterr().out == ""


def test_null_reporter_track_advances_without_error():
    r = NullReporter()
    with r.track("ASR", total=10) as t:
        t.advance()
        t.advance(3)
    # no exception, nothing to assert beyond that


def test_null_reporter_info_and_summary_are_noops(capsys):
    r = NullReporter()
    r.info("hello")
    r.summary(Summary(duration_s=1.0, n_segments=0, speakers=[]))
    assert capsys.readouterr().out == ""


def test_rich_reporter_stage_writes_to_the_given_stream():
    buf = io.StringIO()
    r = RichReporter(file=buf, force_terminal=False)
    with r.stage("resample"):
        pass
    assert "resample" in buf.getvalue()


def test_rich_reporter_non_tty_output_has_no_ansi():
    buf = io.StringIO()
    r = RichReporter(file=buf, force_terminal=False)
    with r.stage("VAD"):
        pass
    r.info("a line")
    assert not ANSI.search(buf.getvalue())


def test_rich_reporter_verbose_gates_info():
    buf = io.StringIO()
    quiet = RichReporter(file=buf, force_terminal=False, verbose=False)
    quiet.info("secret")
    assert "secret" not in buf.getvalue()

    buf2 = io.StringIO()
    loud = RichReporter(file=buf2, force_terminal=False, verbose=True)
    loud.info("shown")
    assert "shown" in buf2.getvalue()


def test_rms_to_meter_full_at_zero_db():
    assert rms_to_meter(0.0, width=8).count("▇") == 8


def test_rms_to_meter_empty_at_silence():
    assert set(rms_to_meter(float("-inf"), width=8)) == {"▁"}
    assert set(rms_to_meter(-80.0, width=8)) == {"▁"}


def test_rms_to_meter_monotonic_and_width():
    quiet = rms_to_meter(-40.0, 8).count("▇")
    loud = rms_to_meter(-10.0, 8).count("▇")
    assert loud >= quiet
    assert len(rms_to_meter(-25.0, 8)) == 8


def test_null_reporter_supports_no_meters_and_meters_is_noop():
    r = NullReporter()
    assert r.supports_meters() is False
    with r.meters({1: "mic", 2: "system"}) as m:
        m.update(1, -12.0)  # no error, no output


def test_fmt_duration_formats_hours_minutes_seconds():
    assert _fmt_duration(3600.0) == "1h 0m 0s"
    assert _fmt_duration(3672.0) == "1h 1m 12s"
    assert _fmt_duration(72.0) == "1m 12s"
    assert _fmt_duration(5.0) == "5s"


def test_rich_reporter_tty_branches_execute():
    # force_terminal=True exercises the status spinner, Progress bar and Live meters paths.
    buf = io.StringIO()
    r = RichReporter(file=buf, force_terminal=True)
    assert r.supports_meters() is True
    with r.stage("work"):
        pass
    with r.track("ASR", total=2) as t:
        t.advance()
        t.advance()
    with r.meters({1: "mic", 2: "system"}) as m:
        m.update(1, -10.0)
        m.update(2, float("-inf"))
    assert buf.getvalue() != ""  # something was rendered, no exception raised


def test_rich_reporter_summary_renders_speaker_table():
    buf = io.StringIO()
    r = RichReporter(file=buf, force_terminal=False)
    r.summary(
        Summary(
            duration_s=3600.0,
            n_segments=41,
            speakers=[("me", 1380.0, "mic"), ("spk_0", 1140.0, "system")],
        )
    )
    out = buf.getvalue()
    assert "me" in out and "spk_0" in out and "41" in out
