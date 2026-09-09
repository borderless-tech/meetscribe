"""Config foundation: XDG paths, TOML loading, and flag>env>config>default resolvers.

Tests never touch the real home: XDG_CONFIG_HOME / XDG_DATA_HOME / MEETSCRIBE_CONFIG
(and HOME, for the fallback paths) are always monkeypatched to tmp_path.
"""

from pathlib import Path

import pytest

from meetscribe import config
from meetscribe.config import ConfigError, Resolved


# ---------------------------------------------------------------- XDG helpers


def test_config_home_honors_xdg_env(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-cfg"))
    assert config.config_home() == tmp_path / "xdg-cfg"


def test_config_home_falls_back_to_dot_config(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.config_home() == tmp_path / ".config"


def test_config_home_ignores_empty_xdg_env(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.config_home() == tmp_path / ".config"


def test_data_home_honors_xdg_env(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    assert config.data_home() == tmp_path / "xdg-data"


def test_data_home_falls_back_to_local_share(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.data_home() == tmp_path / ".local" / "share"


# ------------------------------------------------------- config_path override


def test_config_path_default_location(monkeypatch, tmp_path):
    monkeypatch.delenv("MEETSCRIBE_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config.config_path() == tmp_path / "meetscribe" / "config.toml"


def test_config_path_env_override_wins(monkeypatch, tmp_path):
    override = tmp_path / "elsewhere" / "my.toml"
    monkeypatch.setenv("MEETSCRIBE_CONFIG", str(override))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "unused"))
    assert config.config_path() == override


# -------------------------------------------------------------------- load()


def test_load_missing_file_is_empty_dict(tmp_path):
    assert config.load(tmp_path / "nope.toml") == {}


def test_load_parses_toml_as_is(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[stt]\nbackend = "deepgram"\n', encoding="utf-8")
    assert config.load(p) == {"stt": {"backend": "deepgram"}}


def test_load_default_path_uses_meetscribe_config_env(monkeypatch, tmp_path):
    p = tmp_path / "override.toml"
    p.write_text('[stt]\nlanguage = "en"\n', encoding="utf-8")
    monkeypatch.setenv("MEETSCRIBE_CONFIG", str(p))
    assert config.load() == {"stt": {"language": "en"}}


def test_load_malformed_raises_configerror_with_file_and_line(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[stt]\nbackend = "deepgram\n', encoding="utf-8")  # unclosed string
    with pytest.raises(ConfigError) as exc:
        config.load(p)
    err = exc.value
    assert err.path == str(p)
    assert err.line == 2
    assert str(p) in str(err)
    assert "2" in str(err)


def test_load_unreadable_file_raises_configerror_not_traceback(tmp_path):
    # PermissionError must become ConfigError (exit-2 material with the path) —
    # an uncaught OSError would traceback out of every subcommand.
    p = tmp_path / "config.toml"
    p.write_text("[stt]\n", encoding="utf-8")
    p.chmod(0o000)
    try:
        with pytest.raises(ConfigError) as exc:
            config.load(p)
        assert str(p) in str(exc.value)
    finally:
        p.chmod(0o600)


def test_load_directory_path_raises_configerror(tmp_path):
    # MEETSCRIBE_CONFIG pointing at a directory: IsADirectoryError → ConfigError.
    d = tmp_path / "confdir"
    d.mkdir()
    with pytest.raises(ConfigError) as exc:
        config.load(d)
    assert str(d) in str(exc.value)


def test_load_template_is_valid_toml_and_sets_nothing(tmp_path):
    # All template keys ship commented out: a fresh `config init` must behave
    # exactly like no config file — origin "default" everywhere until edited.
    p = tmp_path / "config.toml"
    p.write_text(config.TEMPLATE, encoding="utf-8")
    cfg = config.load(p)
    assert cfg == {}
    assert config.backend(None, {}, cfg).origin == "default"
    assert config.bundle(None, {}, cfg).origin == "default"


# ----------------------------------------------------------- resolver: backend


def test_backend_default(tmp_path):
    assert config.backend(None, {}, {}) == Resolved("local", "default")


def test_backend_from_config():
    cfg = {"stt": {"backend": "deepgram"}}
    assert config.backend(None, {}, cfg) == Resolved("deepgram", "config")


def test_backend_env_beats_config():
    cfg = {"stt": {"backend": "deepgram"}}
    env = {"STT_BACKEND": "local"}
    assert config.backend(None, env, cfg) == Resolved("local", "env")


def test_backend_flag_beats_env_and_config():
    cfg = {"stt": {"backend": "deepgram"}}
    env = {"STT_BACKEND": "deepgram"}
    assert config.backend("local", env, cfg) == Resolved("local", "flag")


def test_backend_normalizes_case_and_whitespace():
    assert config.backend("  Deepgram ", {}, {}) == Resolved("deepgram", "flag")
    assert config.backend(None, {"STT_BACKEND": " LOCAL "}, {}) == Resolved(
        "local", "env"
    )


def test_backend_empty_layers_fall_through():
    cfg = {"stt": {"backend": ""}}  # template ships empty strings — not "set"
    env = {"STT_BACKEND": ""}
    assert config.backend("", env, cfg) == Resolved("local", "default")


def test_backend_non_string_config_is_error():
    with pytest.raises(ConfigError):
        config.backend(None, {}, {"stt": {"backend": True}})


# ---------------------------------------------------------- resolver: language


def test_language_default():
    assert config.language(None, {}, {}) == Resolved("de", "default")


def test_language_from_config():
    cfg = {"stt": {"language": "en"}}
    assert config.language(None, {}, cfg) == Resolved("en", "config")


def test_language_env_beats_config():
    cfg = {"stt": {"language": "en"}}
    assert config.language(None, {"STT_LANGUAGE": "fr"}, cfg) == Resolved("fr", "env")


def test_language_flag_beats_env_and_config():
    cfg = {"stt": {"language": "en"}}
    env = {"STT_LANGUAGE": "fr"}
    assert config.language("de", env, cfg) == Resolved("de", "flag")


def test_language_empty_config_falls_through():
    assert config.language(None, {}, {"stt": {"language": " "}}) == Resolved(
        "de", "default"
    )


# ----------------------------------------------------------- resolver: api_key


def test_api_key_default_is_none():
    assert config.api_key(None, {}, {}) == Resolved(None, "default")


def test_api_key_from_config():
    cfg = {"deepgram": {"api_key": "dg_secret"}}
    assert config.api_key(None, {}, cfg) == Resolved("dg_secret", "config")


def test_api_key_env_beats_config():
    cfg = {"deepgram": {"api_key": "dg_cfg"}}
    env = {"DEEPGRAM_API_KEY": "dg_env"}
    assert config.api_key(None, env, cfg) == Resolved("dg_env", "env")


def test_api_key_flag_beats_env_and_config():
    cfg = {"deepgram": {"api_key": "dg_cfg"}}
    env = {"DEEPGRAM_API_KEY": "dg_env"}
    assert config.api_key("dg_flag", env, cfg) == Resolved("dg_flag", "flag")


def test_api_key_empty_config_string_is_unset():
    # the shipped template has api_key = "" — must not count as configured
    cfg = {"deepgram": {"api_key": ""}}
    assert config.api_key(None, {}, cfg) == Resolved(None, "default")


# ------------------------------------------------------ resolver: meetings_dir


def test_meetings_dir_default_under_data_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert config.meetings_dir(None, {}, {}) == Resolved(
        tmp_path / "data" / "meetscribe" / "meetings", "default"
    )


def test_meetings_dir_from_config_expands_user(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = {"storage": {"meetings_dir": "~/recordings"}}
    assert config.meetings_dir(None, {}, cfg) == Resolved(
        tmp_path / "recordings", "config"
    )


def test_meetings_dir_flag_beats_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = {"storage": {"meetings_dir": "/cfg/dir"}}
    assert config.meetings_dir("~/flagged", {}, cfg) == Resolved(
        tmp_path / "flagged", "flag"
    )


def test_meetings_dir_empty_config_string_is_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    cfg = {"storage": {"meetings_dir": ""}}
    assert config.meetings_dir(None, {}, cfg) == Resolved(
        tmp_path / "data" / "meetscribe" / "meetings", "default"
    )


# ---------------------------------------------------- resolver: system_source


def test_system_source_default_is_none():
    assert config.system_source(None, {}, {}) == Resolved(None, "default")


def test_system_source_from_config():
    cfg = {"record": {"system_source": "alsa_output.foo.monitor"}}
    assert config.system_source(None, {}, cfg) == Resolved(
        "alsa_output.foo.monitor", "config"
    )


def test_system_source_flag_beats_config():
    cfg = {"record": {"system_source": "cfg.monitor"}}
    assert config.system_source("flag.monitor", {}, cfg) == Resolved(
        "flag.monitor", "flag"
    )


def test_system_source_empty_config_string_is_unset():
    cfg = {"record": {"system_source": ""}}
    assert config.system_source(None, {}, cfg) == Resolved(None, "default")


# --------------------------------------------------- resolvers: bundle/cleanup


@pytest.mark.parametrize("resolver", [config.bundle, config.cleanup])
def test_bool_default_is_true(resolver):
    assert resolver(None, {}, {}) == Resolved(True, "default")


def test_bundle_from_config():
    assert config.bundle(None, {}, {"output": {"bundle": False}}) == Resolved(
        False, "config"
    )


def test_cleanup_from_config():
    assert config.cleanup(None, {}, {"output": {"cleanup": False}}) == Resolved(
        False, "config"
    )


def test_bundle_flag_beats_config():
    cfg = {"output": {"bundle": True}}
    assert config.bundle(False, {}, cfg) == Resolved(False, "flag")
    assert config.bundle(True, {}, {"output": {"bundle": False}}) == Resolved(
        True, "flag"
    )


def test_cleanup_flag_beats_config():
    cfg = {"output": {"cleanup": True}}
    assert config.cleanup(False, {}, cfg) == Resolved(False, "flag")


@pytest.mark.parametrize("bad", ["false", "true", "no", 1, 0])
def test_bundle_config_string_or_int_is_error_not_truthiness(bad):
    with pytest.raises(ConfigError):
        config.bundle(None, {}, {"output": {"bundle": bad}})


@pytest.mark.parametrize("bad", ["false", 1])
def test_cleanup_config_string_or_int_is_error_not_truthiness(bad):
    with pytest.raises(ConfigError):
        config.cleanup(None, {}, {"output": {"cleanup": bad}})


# ---------------------------------------------------------------- unknown_keys


def test_unknown_keys_empty_config():
    assert config.unknown_keys({}) == []


def test_unknown_keys_template_is_fully_known(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(config.TEMPLATE, encoding="utf-8")
    assert config.unknown_keys(config.load(p)) == []


def test_unknown_keys_reports_typoed_key_dotted():
    cfg = {"output": {"bundel": True, "bundle": True}}
    assert config.unknown_keys(cfg) == ["output.bundel"]


def test_unknown_keys_reports_unknown_section():
    cfg = {"outputs": {"bundle": True}}
    assert config.unknown_keys(cfg) == ["outputs"]


def test_unknown_keys_reports_stray_top_level_key():
    assert config.unknown_keys({"backend": "local"}) == ["backend"]


def test_unknown_keys_accepts_known_bk_keys():
    # [bk] stopped being reserved in Phase 2 — its keys are real schema now.
    cfg = {"bk": {"base_url": "https://example.invalid", "token": "t"}}
    assert config.unknown_keys(cfg) == []


def test_unknown_keys_reports_typoed_bk_key():
    # No longer reserved: unknown bk.* keys must warn like any other section.
    cfg = {"bk": {"tokn": "t"}}
    assert config.unknown_keys(cfg) == ["bk.tokn"]


def test_unknown_keys_multiple_sorted_by_appearance():
    cfg = {
        "stt": {"backend": "local", "langauge": "de"},
        "extra": {"x": 1},
    }
    assert config.unknown_keys(cfg) == ["stt.langauge", "extra"]


# -------------------------------------------------------------------- validate


def test_validate_passes_on_template(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(config.TEMPLATE, encoding="utf-8")
    config.validate(config.load(p))  # must not raise


def test_validate_raises_on_wrongly_typed_bool():
    # valid TOML that every run would reject at resolve time — validate must
    # surface it (doctor turns this into a red check).
    with pytest.raises(ConfigError, match="bundle"):
        config.validate({"output": {"bundle": "false"}})


def test_validate_raises_on_wrongly_typed_string():
    with pytest.raises(ConfigError, match="language"):
        config.validate({"stt": {"language": 5}})


def test_validate_raises_on_non_string_api_key():
    with pytest.raises(ConfigError, match="api_key"):
        config.validate({"deepgram": {"api_key": 5}})


# --------------------------------------------------------------- load_warnings


def test_load_warnings_empty_for_clean_config(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[output]\nbundle = false\n", encoding="utf-8")
    p.chmod(0o600)
    assert config.load_warnings(config.load(p), p) == []


def test_load_warnings_reports_unknown_keys(tmp_path):
    # the design's load-time typo detection: warn, then continue.
    p = tmp_path / "config.toml"
    p.write_text("[output]\nbundel = false\n", encoding="utf-8")
    msgs = config.load_warnings(config.load(p), p)
    assert any("output.bundel" in m for m in msgs)


def test_load_warnings_reports_insecure_api_key_perms(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[deepgram]\napi_key = "dg_secret"\n', encoding="utf-8")
    p.chmod(0o644)
    msgs = config.load_warnings(config.load(p), p)
    assert any("0600" in m for m in msgs)
    assert not any("dg_secret" in m for m in msgs)  # never echo the secret


def test_load_warnings_missing_file_is_silent(tmp_path):
    assert config.load_warnings({}, tmp_path / "nope.toml") == []


# ------------------------------------------------------ insecure_api_key_perms


def _write_cfg(tmp_path, body, mode):
    p = tmp_path / "config.toml"
    p.write_text(body, encoding="utf-8")
    p.chmod(mode)
    return p


def test_insecure_perms_true_when_key_set_and_group_other_readable(tmp_path):
    p = _write_cfg(tmp_path, '[deepgram]\napi_key = "dg_secret"\n', 0o644)
    assert config.insecure_api_key_perms(p) is True


def test_insecure_perms_false_when_mode_0600(tmp_path):
    p = _write_cfg(tmp_path, '[deepgram]\napi_key = "dg_secret"\n', 0o600)
    assert config.insecure_api_key_perms(p) is False


def test_insecure_perms_false_when_no_api_key(tmp_path):
    p = _write_cfg(tmp_path, '[deepgram]\napi_key = ""\n', 0o644)
    assert config.insecure_api_key_perms(p) is False


def test_insecure_perms_false_when_file_missing(tmp_path):
    assert config.insecure_api_key_perms(tmp_path / "nope.toml") is False


# --------------------------------------------------- api_key_cmd + lazy fetch


def test_api_key_cmd_resolves_to_marker_without_executing(tmp_path):
    canary = tmp_path / "canary"
    cfg = {"deepgram": {"api_key_cmd": f"touch {canary}"}}
    resolved = config.api_key(None, {}, cfg)
    assert resolved == Resolved(config.ApiKeyCmd(f"touch {canary}"), "config")
    assert not canary.exists()  # resolving must NEVER run the command


def test_api_key_env_beats_cmd():
    cfg = {"deepgram": {"api_key_cmd": "echo from-cmd"}}
    env = {"DEEPGRAM_API_KEY": "dg_env"}
    assert config.api_key(None, env, cfg) == Resolved("dg_env", "env")


def test_api_key_and_cmd_both_set_is_config_error():
    # A leftover static key silently shadowing the keyring command is the
    # stale-secret trap — ambiguity fails loudly.
    cfg = {"deepgram": {"api_key": "dg_static", "api_key_cmd": "pass show dg"}}
    with pytest.raises(ConfigError, match="api_key_cmd"):
        config.api_key(None, {}, cfg)


def test_api_key_cmd_empty_string_is_unset():
    cfg = {"deepgram": {"api_key_cmd": "  "}}
    assert config.api_key(None, {}, cfg) == Resolved(None, "default")


def test_api_key_cmd_is_a_known_schema_key():
    assert config.unknown_keys({"deepgram": {"api_key_cmd": "x"}}) == []


def test_validate_does_not_execute_api_key_cmd(tmp_path):
    canary = tmp_path / "canary"
    cfg = {"deepgram": {"api_key_cmd": f"touch {canary}"}}
    config.validate(cfg)
    assert not canary.exists()


def test_fetch_api_key_passes_through_plain_string():
    assert config.fetch_api_key("dg_plain") == "dg_plain"


def test_fetch_api_key_none_stays_none():
    assert config.fetch_api_key(None) is None


def test_fetch_api_key_executes_cmd_and_strips():
    key = config.fetch_api_key(config.ApiKeyCmd("echo '  dg_from_cmd  '"))
    assert key == "dg_from_cmd"


def test_fetch_api_key_failing_cmd_raises_with_stderr():
    with pytest.raises(ConfigError, match="boom"):
        config.fetch_api_key(config.ApiKeyCmd("echo boom >&2; exit 3"))


def test_fetch_api_key_empty_output_raises():
    with pytest.raises(ConfigError, match="empty"):
        config.fetch_api_key(config.ApiKeyCmd("true"))


# ------------------------------------- SecretCmd / fetch_secret (shared helper)


def test_secret_cmd_carries_cmd_and_label():
    marker = config.SecretCmd("pass show bk", "[bk].token_cmd")
    assert marker.cmd == "pass show bk"
    assert marker.label == "[bk].token_cmd"


def test_api_key_cmd_is_a_secret_cmd_with_deepgram_label():
    # Thin alias: same machinery, pre-filled label, still one positional arg.
    marker = config.ApiKeyCmd("pass show deepgram")
    assert isinstance(marker, config.SecretCmd)
    assert marker.cmd == "pass show deepgram"
    assert marker.label == "[deepgram].api_key_cmd"


def test_fetch_api_key_is_fetch_secret():
    assert config.fetch_api_key is config.fetch_secret


def test_fetch_secret_passes_through_plain_string_and_none():
    assert config.fetch_secret("tok_plain") == "tok_plain"
    assert config.fetch_secret(None) is None


def test_fetch_secret_executes_cmd_and_strips():
    marker = config.SecretCmd("echo '  bk_tok  '", "[bk].token_cmd")
    assert config.fetch_secret(marker) == "bk_tok"


def test_fetch_secret_failure_names_the_label():
    marker = config.SecretCmd("echo boom >&2; exit 3", "[bk].token_cmd")
    with pytest.raises(ConfigError, match=r"\[bk\]\.token_cmd") as exc:
        config.fetch_secret(marker)
    assert "boom" in str(exc.value)


def test_fetch_secret_empty_output_names_the_label():
    marker = config.SecretCmd("true", "[bk].token_cmd")
    with pytest.raises(ConfigError, match=r"\[bk\]\.token_cmd.*empty"):
        config.fetch_secret(marker)


# -------------------------------------------------------- resolver: bk_base_url


def test_bk_base_url_default_is_none():
    assert config.bk_base_url(None, {}, {}) == Resolved(None, "default")


def test_bk_base_url_from_config():
    cfg = {"bk": {"base_url": "https://bk.example.com"}}
    assert config.bk_base_url(None, {}, cfg) == Resolved(
        "https://bk.example.com", "config"
    )


def test_bk_base_url_env_beats_config():
    cfg = {"bk": {"base_url": "https://cfg.example.com"}}
    env = {"MEETSCRIBE_BK_URL": "https://env.example.com"}
    assert config.bk_base_url(None, env, cfg) == Resolved(
        "https://env.example.com", "env"
    )


def test_bk_base_url_strips_trailing_slash():
    cfg = {"bk": {"base_url": "https://bk.example.com/"}}
    assert config.bk_base_url(None, {}, cfg).value == "https://bk.example.com"
    env = {"MEETSCRIBE_BK_URL": "https://env.example.com//"}
    assert config.bk_base_url(None, env, {}).value == "https://env.example.com"


def test_bk_base_url_empty_layers_fall_through():
    cfg = {"bk": {"base_url": ""}}
    env = {"MEETSCRIBE_BK_URL": ""}
    assert config.bk_base_url(None, env, cfg) == Resolved(None, "default")


def test_bk_base_url_non_string_config_is_error():
    with pytest.raises(ConfigError, match="base_url"):
        config.bk_base_url(None, {}, {"bk": {"base_url": 5}})


# ----------------------------------------------------------- resolver: bk_token


def test_bk_token_default_is_none():
    assert config.bk_token(None, {}, {}) == Resolved(None, "default")


def test_bk_token_from_config():
    cfg = {"bk": {"token": "bk_secret"}}
    assert config.bk_token(None, {}, cfg) == Resolved("bk_secret", "config")


def test_bk_token_env_beats_config():
    cfg = {"bk": {"token": "bk_cfg"}}
    env = {"MEETSCRIBE_BK_TOKEN": "bk_env"}
    assert config.bk_token(None, env, cfg) == Resolved("bk_env", "env")


def test_bk_token_empty_config_string_is_unset():
    cfg = {"bk": {"token": ""}}
    assert config.bk_token(None, {}, cfg) == Resolved(None, "default")


def test_bk_token_cmd_resolves_to_marker_without_executing(tmp_path):
    canary = tmp_path / "canary"
    cfg = {"bk": {"token_cmd": f"touch {canary}"}}
    resolved = config.bk_token(None, {}, cfg)
    assert resolved == Resolved(
        config.SecretCmd(f"touch {canary}", "[bk].token_cmd"), "config"
    )
    assert not canary.exists()  # resolving must NEVER run the command


def test_bk_token_env_beats_cmd():
    cfg = {"bk": {"token_cmd": "echo from-cmd"}}
    env = {"MEETSCRIBE_BK_TOKEN": "bk_env"}
    assert config.bk_token(None, env, cfg) == Resolved("bk_env", "env")


def test_bk_token_and_cmd_both_set_is_config_error():
    cfg = {"bk": {"token": "bk_static", "token_cmd": "pass show bk"}}
    with pytest.raises(ConfigError, match="token_cmd"):
        config.bk_token(None, {}, cfg)


def test_bk_token_cmd_empty_string_is_unset():
    cfg = {"bk": {"token_cmd": "  "}}
    assert config.bk_token(None, {}, cfg) == Resolved(None, "default")


# ----------------------------------------------------- resolver: bk_auto_upload


def test_bk_auto_upload_default_is_false():
    assert config.bk_auto_upload(None, {}, {}) == Resolved(False, "default")


def test_bk_auto_upload_from_config():
    cfg = {"bk": {"auto_upload": True}}
    assert config.bk_auto_upload(None, {}, cfg) == Resolved(True, "config")


def test_bk_auto_upload_flag_beats_config():
    cfg = {"bk": {"auto_upload": True}}
    assert config.bk_auto_upload(False, {}, cfg) == Resolved(False, "flag")
    assert config.bk_auto_upload(True, {}, {"bk": {"auto_upload": False}}) == Resolved(
        True, "flag"
    )


@pytest.mark.parametrize("bad", ["false", "true", 1, 0])
def test_bk_auto_upload_config_string_or_int_is_error(bad):
    with pytest.raises(ConfigError):
        config.bk_auto_upload(None, {}, {"bk": {"auto_upload": bad}})


# ------------------------------------------------------------ validate covers bk


def test_validate_raises_on_wrongly_typed_bk_auto_upload():
    with pytest.raises(ConfigError, match="auto_upload"):
        config.validate({"bk": {"auto_upload": "true"}})


def test_validate_raises_on_both_bk_token_keys():
    with pytest.raises(ConfigError, match="token_cmd"):
        config.validate({"bk": {"token": "t", "token_cmd": "pass show bk"}})


def test_validate_raises_on_non_string_bk_base_url():
    with pytest.raises(ConfigError, match="base_url"):
        config.validate({"bk": {"base_url": 5}})


def test_validate_does_not_execute_bk_token_cmd(tmp_path):
    canary = tmp_path / "canary"
    config.validate({"bk": {"token_cmd": f"touch {canary}"}})
    assert not canary.exists()


# --------------------------------------------- secret perms cover [bk].token too


def test_insecure_secret_perms_lists_bk_token(tmp_path):
    p = _write_cfg(tmp_path, '[bk]\ntoken = "bk_secret"\n', 0o644)
    assert config.insecure_secret_perms(p) == ["[bk].token"]


def test_insecure_secret_perms_lists_both_secrets(tmp_path):
    p = _write_cfg(
        tmp_path,
        '[deepgram]\napi_key = "dg_secret"\n[bk]\ntoken = "bk_secret"\n',
        0o644,
    )
    assert config.insecure_secret_perms(p) == ["[deepgram].api_key", "[bk].token"]


def test_insecure_secret_perms_empty_when_mode_0600(tmp_path):
    p = _write_cfg(tmp_path, '[bk]\ntoken = "bk_secret"\n', 0o600)
    assert config.insecure_secret_perms(p) == []


def test_insecure_api_key_perms_alias_covers_bk_token(tmp_path):
    # Old name kept as a bool alias; it now flags either secret.
    p = _write_cfg(tmp_path, '[bk]\ntoken = "bk_secret"\n', 0o644)
    assert config.insecure_api_key_perms(p) is True


def test_load_warnings_reports_insecure_bk_token_perms(tmp_path):
    p = _write_cfg(tmp_path, '[bk]\ntoken = "bk_secret"\n', 0o644)
    msgs = config.load_warnings(config.load(p), p)
    assert any("0600" in m for m in msgs)
    assert not any("bk_secret" in m for m in msgs)  # never echo the secret


# ------------------------------------------------- template: all-commented-out


def test_template_parses_to_pure_defaults():
    # The template must not SET anything: every key commented out, so a fresh
    # `config init` reports origin "default" everywhere until the user edits.
    import tomllib

    parsed = tomllib.loads(config.TEMPLATE)
    assert parsed == {}


def test_template_mentions_every_schema_key():
    # Commented-out or not, the template must document the full v1 surface.
    for section, keys in config._SCHEMA.items():
        assert f"[{section}]" in config.TEMPLATE
        for key in keys:
            assert key in config.TEMPLATE
