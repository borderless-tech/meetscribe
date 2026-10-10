"""``RichInteraction`` — scripted keypresses in, returned answers out (``interaction.py``).

All four seams are injected: ``console`` writes to a StringIO, ``readkey``/``read_line``
pop scripted inputs, ``player`` records the snippets it was asked to play. The asserts are
the returned answer objects (and what reached the player seam) — never rendered strings.
``ffplay_argv`` is a pure function and tested as one.
"""

import io
from datetime import date
from pathlib import Path

import pytest
from rich.console import Console

from meetscribe.interaction import (
    Assignment,
    MappingAnswer,
    MappingCandidate,
    MappingReviewView,
    NewPerson,
    OpenWeb,
    Quit,
    RichInteraction,
    RosterEntry,
    Snippet,
    SpeakerAnswers,
    SpeakerReviewView,
    SpeakerView,
    Suggestion,
    ffplay_argv,
    play_snippet,
)

ENTER = "\r"


def rich_interaction(keys, lines=(), player=None):
    """A RichInteraction with every seam scripted; exhausted scripts raise IndexError,
    so a walk that reads more keys than the test provides fails loudly."""
    keys_left = list(keys)
    lines_left = list(lines)
    return RichInteraction(
        console=Console(file=io.StringIO(), force_terminal=False, width=100),
        readkey=lambda: keys_left.pop(0),
        read_line=lambda prompt="": lines_left.pop(0),
        player=player if player is not None else lambda snippet: None,
    )


# Views mirroring the shape of the vendored speaker-annotation contract fixture:
# "me" carries bk's explicit preselection, spk_0 a top high-confidence voice match plus a
# null-confidence roster entry, spk_1 no suggestions at all (and no snippets).
ALICE = Suggestion(
    person_id="p-alice", name="Alice Anders", confidence="high", source="owner",
    preselected=True,
)
ANNA = Suggestion(
    person_id="p-anna", name="Anna Meier", confidence="high", source="voice_match"
)
BEN = Suggestion(person_id="p-ben", name="Ben Otto", confidence=None, source="roster")

SNIP_ME = Snippet(Path("/m/raw/mic.wav"), 0.0, 5.0)
SNIP_A = Snippet(Path("/m/raw/system.wav"), 10.0, 10.0)
SNIP_B = Snippet(Path("/m/raw/system.wav"), 25.0, 4.0)
SNIP_C = Snippet(Path("/m/raw/system.wav"), 30.0, 1.0)

ME = SpeakerView(
    label="me", talk_time_s=300.0, samples=("hello from the mic",),
    suggestions=(ALICE,), snippets=(SNIP_ME,),
)
SPK_0 = SpeakerView(
    label="spk_0", talk_time_s=120.0, samples=("the longest utterance", "a remark"),
    suggestions=(ANNA, BEN), snippets=(SNIP_A, SNIP_B, SNIP_C),
)
SPK_1 = SpeakerView(label="spk_1", talk_time_s=10.0, samples=(), suggestions=(), snippets=())

ROSTER = (
    RosterEntry(person_id="p-anna", email="anna@example.com", name="Anna Meier"),
    RosterEntry(person_id="p-ben", email="ben@example.com", name=None),
)
WEB_URL = "https://bk.example.com/transcripts/wf-1"


def speaker_view(*speakers):
    return SpeakerReviewView(speakers=speakers, roster=ROSTER, web_url=WEB_URL)


# ---- speaker gate: one key per speaker --------------------------------------------------


def test_enter_per_speaker_confirms_the_preselected_set():
    # me → bk's explicit preselection; spk_0 → the top high-confidence suggestion.
    ia = rich_interaction([ENTER, ENTER, "y"])
    answer = ia.review_speakers(speaker_view(ME, SPK_0))
    assert answer == SpeakerAnswers(
        assignments=(
            Assignment(label="me", person_id="p-alice"),
            Assignment(label="spk_0", person_id="p-anna"),
        )
    )


def test_enter_at_the_summary_also_confirms():
    ia = rich_interaction([ENTER, ENTER])
    answer = ia.review_speakers(speaker_view(ME))
    assert answer == SpeakerAnswers(assignments=(Assignment(label="me", person_id="p-alice"),))


def test_enter_without_a_preselection_is_ignored_and_the_key_reread():
    # spk_1 has no suggestions → Enter must be consumed without answering; the
    # following "u" is the real answer.
    ia = rich_interaction([ENTER, "u", "y"])
    answer = ia.review_speakers(speaker_view(SPK_1))
    assert answer == SpeakerAnswers(assignments=(Assignment(label="spk_1"),))


def test_digit_picks_that_suggestion():
    ia = rich_interaction(["2", "y"])
    answer = ia.review_speakers(speaker_view(SPK_0))
    assert answer == SpeakerAnswers(assignments=(Assignment(label="spk_0", person_id="p-ben"),))


def test_out_of_range_digit_is_ignored():
    ia = rich_interaction(["9", "1", "y"])
    answer = ia.review_speakers(speaker_view(SPK_0))
    assert answer == SpeakerAnswers(assignments=(Assignment(label="spk_0", person_id="p-anna"),))


def test_n_creates_a_new_person_from_name_and_email_lines():
    ia = rich_interaction(["n", "y"], lines=["Carla Neu", "carla@example.com"])
    answer = ia.review_speakers(speaker_view(SPK_1))
    assert answer == SpeakerAnswers(
        assignments=(
            Assignment(
                label="spk_1",
                new_person=NewPerson(name="Carla Neu", email="carla@example.com"),
            ),
        )
    )


def test_n_with_empty_email_line_means_no_email():
    ia = rich_interaction(["n", "y"], lines=["Externer Gast", ""])
    answer = ia.review_speakers(speaker_view(SPK_1))
    assert answer == SpeakerAnswers(
        assignments=(
            Assignment(label="spk_1", new_person=NewPerson(name="Externer Gast", email=None)),
        )
    )


def test_r_picks_from_the_full_roster_by_number():
    ia = rich_interaction(["r", "2", "y"])
    answer = ia.review_speakers(speaker_view(SPK_1))
    assert answer == SpeakerAnswers(assignments=(Assignment(label="spk_1", person_id="p-ben"),))


def test_q_mid_walk_quits_immediately():
    ia = rich_interaction([ENTER, "q"])
    assert isinstance(ia.review_speakers(speaker_view(ME, SPK_0)), Quit)


def test_w_mid_walk_opens_web_immediately():
    ia = rich_interaction(["w"])
    assert isinstance(ia.review_speakers(speaker_view(ME)), OpenWeb)


def test_q_at_the_summary_quits():
    ia = rich_interaction([ENTER, "q"])
    assert isinstance(ia.review_speakers(speaker_view(ME)), Quit)


# ---- speaker gate: playback -------------------------------------------------------------


def test_p_plays_the_longest_snippet_then_cycles_and_the_answer_is_still_collected():
    played = []
    ia = rich_interaction(["p", "p", "1", "y"], player=lambda s: played.append(s))
    answer = ia.review_speakers(speaker_view(SPK_0))
    assert played == [SNIP_A, SNIP_B]  # cycling: longest first, then the next
    assert answer == SpeakerAnswers(assignments=(Assignment(label="spk_0", person_id="p-anna"),))


def test_p_cycles_back_to_the_first_snippet_after_the_last():
    played = []
    ia = rich_interaction(["p", "p", "p", "p", "u", "y"], player=lambda s: played.append(s))
    ia.review_speakers(speaker_view(SPK_0))
    assert played == [SNIP_A, SNIP_B, SNIP_C, SNIP_A]


def test_p_without_snippets_never_calls_the_player_and_the_walk_continues():
    played = []
    ia = rich_interaction(["p", "u", "y"], player=lambda s: played.append(s))
    answer = ia.review_speakers(speaker_view(SPK_1))
    assert played == []
    assert answer == SpeakerAnswers(assignments=(Assignment(label="spk_1"),))


def test_a_player_failure_message_does_not_abort_the_walk():
    ia = rich_interaction(["p", "1", "y"], player=lambda s: "ffplay not found — install ffmpeg")
    answer = ia.review_speakers(speaker_view(SPK_0))
    assert answer == SpeakerAnswers(assignments=(Assignment(label="spk_0", person_id="p-anna"),))


# ---- speaker gate: summary restart ------------------------------------------------------


def test_e_at_the_summary_restarts_the_walk_and_the_second_walk_wins():
    # First walk confirms Alice, summary "e" backs out, second walk leaves "me" unassigned.
    ia = rich_interaction([ENTER, "e", "u", "y"])
    answer = ia.review_speakers(speaker_view(ME))
    assert answer == SpeakerAnswers(assignments=(Assignment(label="me"),))


# ---- mapping gate -----------------------------------------------------------------------

C_HIGH = MappingCandidate(
    meeting_id="m-weekly", title="Weekly Sync",
    start="2026-09-08T07:00:00.000Z", end="2026-09-08T07:30:00.000Z", match="high",
)
C_LOW = MappingCandidate(
    meeting_id="m-notes", title="Planungsnotizen Q4", start=None, end=None, match="low"
)


def mapping_view(candidates=(C_HIGH, C_LOW), allow_create=True):
    return MappingReviewView(candidates=candidates, allow_create=allow_create, web_url=WEB_URL)


def test_digit_selects_that_candidate():
    ia = rich_interaction(["2"])
    assert ia.review_mapping(mapping_view()) == MappingAnswer(meeting_id="m-notes")


def test_enter_confirms_a_sole_high_candidate():
    ia = rich_interaction([ENTER])
    assert ia.review_mapping(mapping_view()) == MappingAnswer(meeting_id="m-weekly")


def test_enter_is_ignored_when_no_candidate_is_sole_high():
    two_high = (C_HIGH, MappingCandidate("m-other", "Other", None, None, "high"))
    ia = rich_interaction([ENTER, "2"])
    assert ia.review_mapping(mapping_view(candidates=two_high)) == MappingAnswer(
        meeting_id="m-other"
    )


def test_n_creates_with_the_typed_title():
    ia = rich_interaction(["n"], lines=["Kickoff Q4"])
    assert ia.review_mapping(mapping_view()) == MappingAnswer(create_title="Kickoff Q4")


def test_n_with_empty_title_defaults_to_meeting_today():
    ia = rich_interaction(["n"], lines=[""])
    assert ia.review_mapping(mapping_view()) == MappingAnswer(
        create_title=f"Meeting {date.today().isoformat()}"
    )


def test_n_is_ignored_when_create_is_not_allowed():
    # lines is empty: reading a title would raise, so the pass-through is behavioral.
    ia = rich_interaction(["n", "1"])
    assert ia.review_mapping(mapping_view(allow_create=False)) == MappingAnswer(
        meeting_id="m-weekly"
    )


def test_mapping_w_and_q():
    assert isinstance(rich_interaction(["w"]).review_mapping(mapping_view()), OpenWeb)
    assert isinstance(rich_interaction(["q"]).review_mapping(mapping_view()), Quit)


# ---- ffplay playback --------------------------------------------------------------------


def test_ffplay_argv_is_the_exact_documented_command():
    snippet = Snippet(Path("/m/raw/system.wav"), 12.3456, 4.2)
    assert ffplay_argv(snippet) == [
        "ffplay", "-nodisp", "-autoexit", "-loglevel", "error",
        "-ss", "12.346", "-t", "4.200", "/m/raw/system.wav",
    ]


def test_play_snippet_runs_ffplay_with_the_argv():
    calls = []
    snippet = Snippet(Path("/m/raw/mic.wav"), 0.0, 5.0)
    result = play_snippet(snippet, run=lambda argv, **kw: calls.append(argv))
    assert calls == [ffplay_argv(snippet)]
    assert result is None


def test_play_snippet_missing_ffplay_returns_a_message_instead_of_raising():
    def no_ffplay(argv, **kw):
        raise FileNotFoundError("ffplay")

    result = play_snippet(Snippet(Path("/m/raw/mic.wav"), 0.0, 5.0), run=no_ffplay)
    assert result == "ffplay not found — install ffmpeg"
