"""View-model building for the bk review loop (``review.py``).

All derivation here is pure and local: talk time, samples, and snippets come from
``transcript.json`` (written/read via ``output``), suggestions/roster/candidates mirror the
vendored contract fixtures under ``fixtures/bk/`` — never hand-written workflow JSON.
"""

import json
import shutil
from pathlib import Path

import pytest

from meetscribe.interaction import (
    MappingCandidate,
    RosterEntry,
    Snippet,
    Suggestion,
)
from meetscribe.output import write_transcript
from meetscribe.review import build_mapping_view, build_speaker_view
from meetscribe.types import Utterance

FIXTURES = Path(__file__).parent / "fixtures" / "bk"


def _task(fixture_name):
    workflow = json.loads((FIXTURES / fixture_name).read_text())
    return workflow["tasks"][0]


def _utt(start, end, speaker, track, text):
    return Utterance(start=start, end=end, speaker=speaker, track=track, text=text, words=())


def _meeting_dir(tmp_path, utterances, with_raw=True):
    meeting = tmp_path / "meetscribe-20261010-100000"
    meeting.mkdir()
    write_transcript(meeting / "transcript.json", "m-review", 60.0, utterances)
    if with_raw:
        raw = meeting / "raw"
        raw.mkdir()
        (raw / "mic.wav").touch()
        (raw / "system.wav").touch()
    return meeting


# Synthetic transcript matching the speaker-annotation fixture's labels: spk_0 gets
# 10 s + 4 s + 1 s on the system track (deliberately out of duration order in the file),
# "me" 5 s on mic, spk_1 2 s on system; spk_2 is in the payload but NOT in the transcript.
UTTERANCES = [
    _utt(0.0, 5.0, "me", "mic", "hello from the mic track"),
    _utt(25.0, 29.0, "spk_0", "system", "a four second remark"),
    _utt(10.0, 20.0, "spk_0", "system", "the longest utterance spk zero ever made"),
    _utt(30.0, 31.0, "spk_0", "system", "yes"),
    _utt(40.0, 42.0, "spk_1", "system", "short reply"),
]


@pytest.fixture
def speaker_task():
    return _task("workflow-awaiting-review-speaker-annotation.json")


# ---- build_speaker_view: derivation from the transcript -------------------------------


def test_talk_time_sums_all_of_a_speakers_segments(tmp_path, speaker_task):
    meeting = _meeting_dir(tmp_path, UTTERANCES)
    view = build_speaker_view(speaker_task, meeting)
    by_label = {s.label: s for s in view.speakers}
    assert by_label["spk_0"].talk_time_s == 15.0
    assert by_label["me"].talk_time_s == 5.0


def test_samples_are_segment_texts_longest_first(tmp_path, speaker_task):
    meeting = _meeting_dir(tmp_path, UTTERANCES)
    view = build_speaker_view(speaker_task, meeting)
    by_label = {s.label: s for s in view.speakers}
    assert by_label["spk_0"].samples == (
        "the longest utterance spk zero ever made",
        "a four second remark",
        "yes",
    )


def test_long_sample_text_is_truncated_at_a_word_boundary(tmp_path, speaker_task):
    long_text = ("abcdefghij " * 20).strip()  # 20 ten-char words, 219 chars
    meeting = _meeting_dir(tmp_path, [_utt(0.0, 5.0, "me", "mic", long_text)])
    view = build_speaker_view(speaker_task, meeting)
    by_label = {s.label: s for s in view.speakers}
    # Cut at the last word boundary at or before ~120 chars: 10 whole words + ellipsis.
    assert by_label["me"].samples == (("abcdefghij " * 10).strip() + "…",)


def test_snippets_are_longest_first_and_point_at_the_tracks_wav(tmp_path, speaker_task):
    meeting = _meeting_dir(tmp_path, UTTERANCES)
    view = build_speaker_view(speaker_task, meeting)
    by_label = {s.label: s for s in view.speakers}
    assert by_label["spk_0"].snippets == (
        Snippet(meeting / "raw" / "system.wav", 10.0, 10.0),
        Snippet(meeting / "raw" / "system.wav", 25.0, 4.0),
        Snippet(meeting / "raw" / "system.wav", 30.0, 1.0),
    )
    assert by_label["me"].snippets == (Snippet(meeting / "raw" / "mic.wav", 0.0, 5.0),)


def test_snippets_are_capped_at_the_three_longest_segments(tmp_path, speaker_task):
    utts = UTTERANCES + [_utt(50.0, 53.0, "spk_0", "system", "a three second aside")]
    meeting = _meeting_dir(tmp_path, utts)
    view = build_speaker_view(speaker_task, meeting)
    by_label = {s.label: s for s in view.speakers}
    assert by_label["spk_0"].snippets == (
        Snippet(meeting / "raw" / "system.wav", 10.0, 10.0),
        Snippet(meeting / "raw" / "system.wav", 25.0, 4.0),
        Snippet(meeting / "raw" / "system.wav", 50.0, 3.0),
    )


def test_missing_raw_dir_yields_no_snippets_but_everything_else_intact(
    tmp_path, speaker_task
):
    meeting = _meeting_dir(tmp_path, UTTERANCES, with_raw=False)
    view = build_speaker_view(speaker_task, meeting)
    assert all(s.snippets == () for s in view.speakers)
    by_label = {s.label: s for s in view.speakers}
    assert by_label["spk_0"].talk_time_s == 15.0  # derivation unaffected


def test_deleting_raw_after_the_fact_degrades_the_same_way(tmp_path, speaker_task):
    meeting = _meeting_dir(tmp_path, UTTERANCES)
    shutil.rmtree(meeting / "raw")
    view = build_speaker_view(speaker_task, meeting)
    assert all(s.snippets == () for s in view.speakers)


# ---- build_speaker_view: payload mirroring --------------------------------------------


def test_suggestions_mirror_the_payload_including_preselected_and_null_confidence(
    tmp_path, speaker_task
):
    meeting = _meeting_dir(tmp_path, UTTERANCES)
    view = build_speaker_view(speaker_task, meeting)
    by_label = {s.label: s for s in view.speakers}
    assert by_label["me"].suggestions == (
        Suggestion(
            person_id="7c0a9d8e-1f2b-4c3d-8e4f-5a6b7c8d9e0f",
            name="Alice Anders",
            confidence="high",
            source="owner",
            preselected=True,
        ),
    )
    assert by_label["spk_0"].suggestions == (
        Suggestion(
            person_id="3b2c4d5e-6f70-4182-9394-a5b6c7d8e9f0",
            name="Anna Meier",
            confidence="high",
            source="voice_match",
            preselected=False,
        ),
        Suggestion(
            person_id="4c3d5e6f-7081-4293-a4b5-c6d7e8f9a0b1",
            name="Ben Otto",
            confidence=None,  # contract's JSON null
            source="roster",
            preselected=False,
        ),
    )


def test_roster_mirrors_the_payload_including_null_name(tmp_path, speaker_task):
    meeting = _meeting_dir(tmp_path, UTTERANCES)
    view = build_speaker_view(speaker_task, meeting)
    assert view.roster == (
        RosterEntry(
            person_id="3b2c4d5e-6f70-4182-9394-a5b6c7d8e9f0",
            email="anna@example.com",
            name="Anna Meier",
        ),
        RosterEntry(
            person_id="4c3d5e6f-7081-4293-a4b5-c6d7e8f9a0b1",
            email="ben@example.com",
            name=None,
        ),
    )


def test_payload_speaker_absent_from_transcript_still_appears_empty(
    tmp_path, speaker_task
):
    # bk may know speakers we lost locally — the gate must still let the user assign them.
    meeting = _meeting_dir(tmp_path, UTTERANCES)
    view = build_speaker_view(speaker_task, meeting)
    by_label = {s.label: s for s in view.speakers}
    assert by_label["spk_2"].talk_time_s == 0.0
    assert by_label["spk_2"].samples == ()
    assert by_label["spk_2"].snippets == ()


def test_speakers_keep_the_payload_order(tmp_path, speaker_task):
    meeting = _meeting_dir(tmp_path, UTTERANCES)
    view = build_speaker_view(speaker_task, meeting)
    assert tuple(s.label for s in view.speakers) == ("me", "spk_0", "spk_1", "spk_2")


def test_speaker_view_carries_the_tasks_web_url(tmp_path, speaker_task):
    meeting = _meeting_dir(tmp_path, UTTERANCES)
    view = build_speaker_view(speaker_task, meeting)
    assert view.web_url == speaker_task["web_url"]


# ---- build_mapping_view ----------------------------------------------------------------


def test_mapping_view_mirrors_candidates_allow_create_and_web_url():
    task = _task("workflow-awaiting-review-meeting-mapping.json")
    view = build_mapping_view(task)
    assert view.candidates == (
        MappingCandidate(
            meeting_id="1f0e2d3c-4b5a-4968-8776-655443322110",
            title="Weekly Sync",
            start="2026-09-08T07:00:00.000Z",
            end="2026-09-08T07:30:00.000Z",
            match="high",
        ),
        MappingCandidate(
            meeting_id="2a1b3c4d-5e6f-4a80-9b1c-2d3e4f5a6b7c",
            title="Planungsnotizen Q4",
            start=None,  # contract's JSON null
            end=None,
            match="low",
        ),
    )
    assert view.allow_create is True
    assert view.web_url == task["web_url"]
