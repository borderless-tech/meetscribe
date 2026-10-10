"""The review-loop engine (``review.run_review``) and its submission translation.

Everything is driven offline: a ``FakeBkClient`` replays sequences of the vendored
contract fixtures (``fixtures/bk/``), ``ScriptedInteraction`` feeds canned answers, and
injected ``sleep``/``open_url`` seams record the engine's observable side effects. The
contract tests assert FULL submitted bodies against the ``task-submit-*.json`` fixtures —
never fragments.
"""

import copy
import json
from pathlib import Path

from meetscribe.bk import (
    BkAuthError,
    BkError,
    BkInvalidSubmission,
    BkRevisionConflict,
    read_workflow_ref,
    write_workflow_ref,
)
from meetscribe.interaction import (
    Assignment,
    MappingAnswer,
    NewPerson,
    OpenWeb,
    Quit,
    ScriptedInteraction,
    SpeakerAnswers,
)
from meetscribe.output import write_transcript
from meetscribe.review import mapping_submission, run_review, speaker_submission
from meetscribe.types import Utterance

FIXTURES = Path(__file__).parent / "fixtures" / "bk"


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())


# ---- fakes ------------------------------------------------------------------------------


class FakeBkClient:
    """Replays canned ``workflow()``/``submit_task()`` results (dicts or exceptions)."""

    def __init__(self, states, submit_results=()):
        self._states = list(states)
        self._submit_results = list(submit_results)
        self.workflow_calls = 0
        self.workflow_ids = []  # the id each poll asked for, in order
        self.submitted = []  # (submit_url, body) in order

    def workflow(self, workflow_id):
        self.workflow_calls += 1
        self.workflow_ids.append(workflow_id)
        item = self._states.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def submit_task(self, submit_url, body):
        self.submitted.append((submit_url, body))
        item = self._submit_results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class RecordingSleep:
    def __init__(self, raise_keyboard_interrupt_on=None):
        self.calls = []
        self._raise_on = raise_keyboard_interrupt_on

    def __call__(self, seconds):
        self.calls.append(seconds)
        if self._raise_on is not None and len(self.calls) == self._raise_on:
            raise KeyboardInterrupt


class RecordingReporter:
    def __init__(self):
        self.warnings = []
        self.stages = []  # ("enter"/"exit", label) events, in order

    def info(self, msg):
        pass

    def warn(self, msg):
        self.warnings.append(msg)

    def stage(self, label):
        events = self.stages

        class _Stage:
            def __enter__(self):
                events.append(("enter", label))

            def __exit__(self, *exc):
                events.append(("exit", label))

        return _Stage()


class RecordingOpen:
    def __init__(self):
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        return True


SEEDED_CHECKED_AT = "2026-10-09T08:00:00+00:00"
WF_ID = "wf-11111111-1111-4111-8111-111111111111"

# Minimal transcript backing the speaker gate (spk_2 stays payload-only, as in
# test_review_views): derivation itself is covered there, the engine only needs
# build_speaker_view to succeed on a real meeting dir.
UTTERANCES = [
    Utterance(start=0.0, end=5.0, speaker="me", track="mic", text="hello", words=()),
    Utterance(start=10.0, end=20.0, speaker="spk_0", track="system", text="longest", words=()),
    Utterance(start=40.0, end=42.0, speaker="spk_1", track="system", text="reply", words=()),
]


def _meeting_dir(tmp_path, with_transcript=False):
    meeting = tmp_path / "meetscribe-20261010-100000"
    meeting.mkdir()
    write_workflow_ref(
        meeting,
        {
            "workflow_id": WF_ID,
            "state_url": f"/api/meetscribe/v1/workflows/{WF_ID}",
            "web_url": f"https://bk.example.com/transcripts/{WF_ID}",
            "uploaded_at": "2026-10-09T07:00:00+00:00",
            "state": "processing",
            "checked_at": SEEDED_CHECKED_AT,
        },
    )
    if with_transcript:
        write_transcript(meeting / "transcript.json", "m-review", 60.0, UTTERANCES)
    return meeting


MAPPING_ANSWER = MappingAnswer(meeting_id="1f0e2d3c-4b5a-4968-8776-655443322110")
SPEAKER_ANSWERS = SpeakerAnswers(
    assignments=(
        Assignment("me", person_id="7c0a9d8e-1f2b-4c3d-8e4f-5a6b7c8d9e0f"),
        Assignment("spk_0", person_id="3b2c4d5e-6f70-4182-9394-a5b6c7d8e9f0"),
        Assignment("spk_1", new_person=NewPerson(name="Externer Gast", email=None)),
        Assignment("spk_2"),
    )
)


# ---- the engine -------------------------------------------------------------------------


def test_happy_path_walks_both_gates_and_submits_the_contract_bodies(tmp_path):
    meeting = _meeting_dir(tmp_path, with_transcript=True)
    mapping_wf = _fixture("workflow-awaiting-review-meeting-mapping.json")
    speakers_wf = _fixture("workflow-awaiting-review-speaker-annotation.json")
    client = FakeBkClient(
        states=[
            _fixture("workflow-processing.json"),
            _fixture("workflow-processing.json"),
            _fixture("workflow-processing.json"),
            mapping_wf,
            speakers_wf,
        ],
        submit_results=[
            _fixture("workflow-processing.json"),  # 200 = full fresh state
            _fixture("workflow-done.json"),
        ],
    )
    interaction = ScriptedInteraction([MAPPING_ANSWER, SPEAKER_ANSWERS])
    sleep = RecordingSleep()
    emitted = []

    code = run_review(
        meeting, client, interaction, sleep=sleep, open_url=RecordingOpen(),
        emit=emitted.append,
    )

    assert code == 0
    # Every poll asked bk for THIS meeting's workflow — the id from the persisted ref.
    assert client.workflow_ids == [WF_ID] * 5
    # THE contract test: both submitted bodies equal the canonical fixtures, and each
    # went to its task's own submit_url.
    assert client.submitted == [
        (
            mapping_wf["tasks"][0]["submit_url"],
            _fixture("task-submit-meeting-mapping-select.json"),
        ),
        (
            speakers_wf["tasks"][0]["submit_url"],
            _fixture("task-submit-speaker-annotation.json"),
        ),
    ]
    # 3 processing polls + 1 after the submit-returned processing state, all at 5 s.
    assert sleep.calls == [5.0, 5.0, 5.0, 5.0]
    ref = read_workflow_ref(meeting)
    assert ref["state"] == "done"
    assert ref["checked_at"] != SEEDED_CHECKED_AT
    # The single machine-useful stdout line (design §gate-rendering).
    assert emitted == [f"{WF_ID} done"]


def test_submit_response_is_used_directly_as_the_next_state(tmp_path):
    # One GET brings the gate; the submit's 200 body (done) feeds the machine — no
    # extra fetch, no sleeping.
    meeting = _meeting_dir(tmp_path)
    client = FakeBkClient(
        states=[_fixture("workflow-awaiting-review-meeting-mapping.json")],
        submit_results=[_fixture("workflow-done.json")],
    )
    sleep = RecordingSleep()

    code = run_review(
        meeting, client, ScriptedInteraction([MAPPING_ANSWER]), sleep=sleep,
        open_url=RecordingOpen(),
    )

    assert code == 0
    assert client.workflow_calls == 1
    assert sleep.calls == []


def test_poll_cadence_backs_off_to_15s_after_the_first_minute(tmp_path):
    meeting = _meeting_dir(tmp_path)
    client = FakeBkClient(
        states=[_fixture("workflow-processing.json")] * 14 + [_fixture("workflow-done.json")]
    )
    sleep = RecordingSleep()

    code = run_review(
        meeting, client, ScriptedInteraction([]), sleep=sleep, open_url=RecordingOpen()
    )

    assert code == 0
    # 12 × 5 s = the first minute; from then on 15 s.
    assert sleep.calls == [5.0] * 12 + [15.0] * 2


def test_409_refetches_and_rerenders_the_gate_with_the_fresh_revision(tmp_path):
    meeting = _meeting_dir(tmp_path)
    stale = _fixture("workflow-awaiting-review-meeting-mapping.json")
    fresh = copy.deepcopy(stale)
    fresh["tasks"][0]["revision"] = 2
    fresh["tasks"][0]["payload"]["candidates"][0]["title"] = "Weekly Sync (moved)"
    client = FakeBkClient(
        states=[stale, fresh],
        submit_results=[
            BkRevisionConflict("the task changed"),
            _fixture("workflow-done.json"),
        ],
    )
    interaction = ScriptedInteraction([MAPPING_ANSWER, MAPPING_ANSWER])

    code = run_review(
        meeting, client, interaction, sleep=RecordingSleep(), open_url=RecordingOpen()
    )

    assert code == 0
    # The gate was re-rendered from the re-fetched task: the second view carries the
    # fresh payload, and the second body echoes the fresh revision.
    assert [v.candidates[0].title for v in interaction.seen] == [
        "Weekly Sync",
        "Weekly Sync (moved)",
    ]
    assert client.submitted[1][1] == {
        "revision": 2,
        "meeting_id": "1f0e2d3c-4b5a-4968-8776-655443322110",
    }


def test_422_notifies_the_server_message_and_reasks_the_same_gate(tmp_path):
    meeting = _meeting_dir(tmp_path)
    client = FakeBkClient(
        states=[_fixture("workflow-awaiting-review-meeting-mapping.json")],
        submit_results=[
            BkInvalidSubmission("title too long"),
            _fixture("workflow-done.json"),
        ],
    )
    interaction = ScriptedInteraction(
        [MappingAnswer(create_title="x" * 500), MAPPING_ANSWER]
    )

    code = run_review(
        meeting, client, interaction, sleep=RecordingSleep(), open_url=RecordingOpen()
    )

    assert code == 0
    assert interaction.messages == ["title too long"]
    # Same gate, same payload, asked twice; the second answer went out.
    assert len(interaction.seen) == 2
    assert interaction.seen[0] == interaction.seen[1]
    assert client.submitted[1][1] == _fixture("task-submit-meeting-mapping-select.json")


def test_unknown_task_type_falls_back_to_the_web_ui(tmp_path):
    meeting = _meeting_dir(tmp_path)
    wf = _fixture("workflow-awaiting-review-meeting-mapping.json")
    wf["tasks"][0]["type"] = "zz_new"
    client = FakeBkClient(states=[wf])
    interaction = ScriptedInteraction([])
    open_url = RecordingOpen()

    code = run_review(
        meeting, client, interaction, sleep=RecordingSleep(), open_url=open_url
    )

    assert code == 0
    assert open_url.urls == [wf["tasks"][0]["web_url"]]
    assert client.submitted == []
    assert interaction.seen == []  # never asked a gate it cannot render


def test_openweb_answer_opens_the_tasks_web_url_and_submits_nothing(tmp_path):
    meeting = _meeting_dir(tmp_path)
    wf = _fixture("workflow-awaiting-review-meeting-mapping.json")
    client = FakeBkClient(states=[wf])
    open_url = RecordingOpen()

    code = run_review(
        meeting, client, ScriptedInteraction([OpenWeb()]), sleep=RecordingSleep(),
        open_url=open_url,
    )

    assert code == 0
    assert open_url.urls == [wf["tasks"][0]["web_url"]]
    assert client.submitted == []


def test_quit_answer_exits_cleanly_without_submitting_or_opening(tmp_path):
    meeting = _meeting_dir(tmp_path)
    client = FakeBkClient(
        states=[_fixture("workflow-awaiting-review-meeting-mapping.json")]
    )
    open_url = RecordingOpen()
    emitted = []

    code = run_review(
        meeting, client, ScriptedInteraction([Quit()]), sleep=RecordingSleep(),
        open_url=open_url, emit=emitted.append,
    )

    assert code == 0
    assert client.submitted == []
    assert open_url.urls == []
    # Quitting still leaves the machine-usable outcome on stdout.
    assert emitted == [f"{WF_ID} awaiting_review"]


def test_failed_workflow_exits_1(tmp_path):
    meeting = _meeting_dir(tmp_path)
    client = FakeBkClient(states=[_fixture("workflow-failed.json")])
    emitted = []

    code = run_review(
        meeting, client, ScriptedInteraction([]), sleep=RecordingSleep(),
        open_url=RecordingOpen(), emit=emitted.append,
    )

    assert code == 1
    assert read_workflow_ref(meeting)["state"] == "failed"
    assert emitted == [f"{WF_ID} failed"]


def test_done_immediately_exits_0_and_persists_the_state(tmp_path):
    meeting = _meeting_dir(tmp_path)
    client = FakeBkClient(states=[_fixture("workflow-done.json")])

    code = run_review(
        meeting, client, ScriptedInteraction([]), sleep=RecordingSleep(),
        open_url=RecordingOpen(),
    )

    assert code == 0
    ref = read_workflow_ref(meeting)
    assert ref["state"] == "done"
    assert ref["checked_at"] != SEEDED_CHECKED_AT


def test_one_polling_blip_warns_and_keeps_polling(tmp_path):
    meeting = _meeting_dir(tmp_path)
    client = FakeBkClient(
        states=[
            _fixture("workflow-processing.json"),
            BkError("connection reset"),
            _fixture("workflow-processing.json"),
            _fixture("workflow-done.json"),
        ]
    )
    reporter = RecordingReporter()

    code = run_review(
        meeting, client, ScriptedInteraction([]), reporter, sleep=RecordingSleep(),
        open_url=RecordingOpen(),
    )

    assert code == 0
    assert client.workflow_calls == 4  # the blip did not end the session
    assert len(reporter.warnings) == 1


def test_five_consecutive_poll_failures_abort_with_exit_1(tmp_path):
    meeting = _meeting_dir(tmp_path)
    client = FakeBkClient(states=[BkError("down")] * 5)

    code = run_review(
        meeting, client, ScriptedInteraction([]), RecordingReporter(),
        sleep=RecordingSleep(), open_url=RecordingOpen(),
    )

    assert code == 1
    assert client.workflow_calls == 5


def test_auth_error_while_polling_exits_2_without_retrying(tmp_path):
    # Contract rule: "401 → actionable client error, never a retry loop" — a revoked
    # token must not be re-sent, and the message (naming the token knobs) must surface.
    meeting = _meeting_dir(tmp_path)
    message = "bk rejected the request (HTTP 401) — check [bk].token"
    client = FakeBkClient(states=[BkAuthError(message)] * 5)
    interaction = ScriptedInteraction([])
    sleep = RecordingSleep()

    code = run_review(
        meeting, client, interaction, RecordingReporter(), sleep=sleep,
        open_url=RecordingOpen(),
    )

    assert code == 2
    assert client.workflow_calls == 1  # never re-sent the known-bad token
    assert sleep.calls == []
    assert interaction.messages == [message]


def test_auth_error_during_submit_exits_2_without_reasking(tmp_path):
    # A 401 at submit time is the same token problem: surfacing it beats re-asking
    # the human for answers bk will reject again.
    meeting = _meeting_dir(tmp_path)
    message = "bk rejected the request (HTTP 401) — check [bk].token"
    client = FakeBkClient(
        states=[_fixture("workflow-awaiting-review-meeting-mapping.json")],
        submit_results=[BkAuthError(message)],
    )
    interaction = ScriptedInteraction([MAPPING_ANSWER])

    code = run_review(
        meeting, client, interaction, sleep=RecordingSleep(), open_url=RecordingOpen()
    )

    assert code == 2
    assert len(client.submitted) == 1
    assert len(interaction.seen) == 1  # the gate was NOT re-asked
    assert interaction.messages == [message]


def test_submit_network_blip_offers_retry_and_resends_the_same_body(tmp_path):
    # Design §errors: the answers are still in memory, so a transport blip offers a
    # retry of the SAME body instead of re-walking the whole gate.
    meeting = _meeting_dir(tmp_path)
    client = FakeBkClient(
        states=[_fixture("workflow-awaiting-review-meeting-mapping.json")],
        submit_results=[BkError("connection reset"), _fixture("workflow-done.json")],
    )
    interaction = ScriptedInteraction([MAPPING_ANSWER, True])  # True = retry

    code = run_review(
        meeting, client, interaction, sleep=RecordingSleep(), open_url=RecordingOpen()
    )

    assert code == 0
    body = _fixture("task-submit-meeting-mapping-select.json")
    url = client.submitted[0][0]
    assert client.submitted == [(url, body), (url, body)]  # identical resend
    assert len(interaction.seen) == 1  # the gate was asked exactly once


def test_submit_network_blip_declined_retry_exits_0_without_resending(tmp_path):
    meeting = _meeting_dir(tmp_path)
    client = FakeBkClient(
        states=[_fixture("workflow-awaiting-review-meeting-mapping.json")],
        submit_results=[BkError("connection reset")],
    )
    interaction = ScriptedInteraction([MAPPING_ANSWER, False])  # False = quit

    code = run_review(
        meeting, client, interaction, sleep=RecordingSleep(), open_url=RecordingOpen()
    )

    assert code == 0
    assert len(client.submitted) == 1


def test_processing_stint_renders_one_reporter_stage_span(tmp_path):
    # Design §engine: "spinner via Reporter" — one stage spans the whole processing
    # stint (not one per poll), and it is closed before the loop returns.
    meeting = _meeting_dir(tmp_path)
    client = FakeBkClient(
        states=[_fixture("workflow-processing.json")] * 3 + [_fixture("workflow-done.json")]
    )
    reporter = RecordingReporter()

    code = run_review(
        meeting, client, ScriptedInteraction([]), reporter, sleep=RecordingSleep(),
        open_url=RecordingOpen(),
    )

    assert code == 0
    assert [event for event, _ in reporter.stages] == ["enter", "exit"]


def test_keyboard_interrupt_during_sleep_exits_0_with_the_last_state_persisted(tmp_path):
    meeting = _meeting_dir(tmp_path)
    client = FakeBkClient(states=[_fixture("workflow-processing.json")] * 2)
    sleep = RecordingSleep(raise_keyboard_interrupt_on=2)

    code = run_review(
        meeting, client, ScriptedInteraction([]), sleep=sleep, open_url=RecordingOpen()
    )

    assert code == 0
    ref = read_workflow_ref(meeting)
    assert ref["state"] == "processing"
    assert ref["checked_at"] != SEEDED_CHECKED_AT


# ---- submission translation: pure functions vs. the contract fixtures ------------------


def test_mapping_submission_select_matches_the_contract_fixture():
    answer = MappingAnswer(meeting_id="1f0e2d3c-4b5a-4968-8776-655443322110")
    assert mapping_submission(answer, 1) == _fixture(
        "task-submit-meeting-mapping-select.json"
    )


def test_mapping_submission_create_matches_the_contract_fixture():
    answer = MappingAnswer(create_title="Spontanes Gespräch mit Anna")
    assert mapping_submission(answer, 1) == _fixture(
        "task-submit-meeting-mapping-create.json"
    )


def test_speaker_submission_matches_the_contract_fixture():
    # Covers all three assignment forms: person_id, person-create, and unassigned
    # (the contract's explicit ``"person_id": null``).
    answers = SpeakerAnswers(
        assignments=(
            Assignment("me", person_id="7c0a9d8e-1f2b-4c3d-8e4f-5a6b7c8d9e0f"),
            Assignment("spk_0", person_id="3b2c4d5e-6f70-4182-9394-a5b6c7d8e9f0"),
            Assignment("spk_1", new_person=NewPerson(name="Externer Gast", email=None)),
            Assignment("spk_2"),
        )
    )
    assert speaker_submission(answers, 3) == _fixture(
        "task-submit-speaker-annotation.json"
    )
