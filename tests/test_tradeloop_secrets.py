"""Keys from an env file: only the ones the pipeline reads, under the names it reads them by."""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

from tests.test_tradeloop_cli import FakeClient
from tradeloop import cli, secrets
from tradeloop.config import LoopConfig
from tradeloop.store import Store

FILE = """\
OPENAI_API_KEY=sk-openai
GITHUB_TOKEN=not-for-the-pipeline
ALPCA_PAPER_TRADING_API_KEY=PKFILE
ALPCA_PAPER_TRADING_SECRET_KEY=file-secret
TYPESAFE_TOKEN=ts-token
FRED_API_KEY=
B2_APPLICATION_KEY=also-not-for-the-pipeline
"""


@pytest.fixture
def env_file(tmp_path):
    path = tmp_path / "tokens.env"
    path.write_text(FILE)
    return path


@pytest.mark.unit
class TestLoadingAnEnvFile:
    def test_known_keys_are_set_under_the_names_the_pipeline_reads(self, env_file):
        environ: dict = {}
        taken = secrets.load_env_file(env_file, environ)
        assert environ == {"OPENAI_API_KEY": "sk-openai", "APCA_API_KEY_ID": "PKFILE",
                           "APCA_API_SECRET_KEY": "file-secret", "TYPESAFE_API_KEY": "ts-token"}
        assert sorted(taken) == sorted(environ)

    def test_nothing_else_in_the_file_reaches_the_environment(self, env_file):
        environ: dict = {}
        secrets.load_env_file(env_file, environ)
        assert "GITHUB_TOKEN" not in environ and "B2_APPLICATION_KEY" not in environ
        assert "TYPESAFE_TOKEN" not in environ          # only under the name that is read

    def test_what_is_already_set_wins(self, env_file):
        environ = {"APCA_API_KEY_ID": "PKENV", "TYPESAFE_API_KEY": "from-env"}
        taken = secrets.load_env_file(env_file, environ)
        assert environ["APCA_API_KEY_ID"] == "PKENV" and environ["TYPESAFE_API_KEY"] == "from-env"
        assert "APCA_API_KEY_ID" not in taken and "APCA_API_SECRET_KEY" in taken

    def test_an_empty_value_is_not_a_key(self, env_file):
        environ: dict = {}
        secrets.load_env_file(env_file, environ)
        assert "FRED_API_KEY" not in environ

    def test_the_canonical_name_in_the_file_wins_over_an_alias(self, tmp_path):
        path = tmp_path / "t.env"
        path.write_text("TYPESAFE_TOKEN=alias\nTYPESAFE_API_KEY=canonical\n")
        environ: dict = {}
        secrets.load_env_file(path, environ)
        assert environ["TYPESAFE_API_KEY"] == "canonical"

    def test_no_file_configured_sets_nothing_and_a_missing_one_is_an_error(self, tmp_path):
        assert secrets.load_env_file(None, {}) == []
        with pytest.raises(ValueError, match="does not exist"):
            secrets.load_env_file(tmp_path / "gone.env", {})

    def test_every_model_providers_key_is_a_known_name(self):
        names = secrets.known_names()
        assert {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "FRED_API_KEY", "TYPESAFE_API_KEY",
                "APCA_API_KEY_ID"} <= set(names)


@pytest.mark.unit
class TestTheCommandsUseIt:
    def test_check_finds_its_keys_through_the_configured_file(self, tmp_path, env_file, monkeypatch):
        monkeypatch.setenv("TRADELOOP_DATA_DIR", str(tmp_path / "data"))
        for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "TYPESAFE_API_KEY", "OPENAI_API_KEY", "FRED_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "config.toml").write_text(f'env_file = "{env_file}"\n')
        seen = []
        monkeypatch.setattr(cli, "AlpacaClient", lambda key, secret: seen.append((key, secret)) or FakeClient())

        result = CliRunner().invoke(cli.app, ["check"])
        assert result.exit_code == 0, result.output
        assert seen == [("PKFILE", "file-secret")]
        assert "optional: TYPESAFE_API_KEY  set" in result.output
        assert "optional: FRED_API_KEY  not set" in result.output
        assert "file-secret" not in result.output and "ts-token" not in result.output    # names, never values

    def test_a_configured_file_that_is_missing_stops_the_command(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADELOOP_DATA_DIR", str(tmp_path))
        (tmp_path / "config.toml").write_text('env_file = "/nowhere/tokens.env"\n')
        result = CliRunner().invoke(cli.app, ["check"])
        assert result.exit_code == 1 and "does not exist" in result.output


@pytest.mark.unit
class TestTheAfterDashboardCommand:
    def _cfg(self, tmp_path, command):
        return LoopConfig(data_dir=tmp_path / "loop", after_dashboard=command)

    def test_it_runs_on_the_dashboard_folder_after_each_build(self, tmp_path):
        marker = tmp_path / "ran.txt"
        script = "import sys, pathlib; pathlib.Path(sys.argv[2]).write_text(sys.argv[1])"
        cfg = self._cfg(tmp_path, [sys.executable, "-c", script, "{dir}", str(marker)])
        with Store(cfg.db_path) as store:
            index = cli.publish_dashboard(cfg, store)
        assert index.is_file() and marker.read_text() == str(cfg.dashboard_dir)

    def test_a_failing_command_is_logged_and_the_pages_are_still_built(self, tmp_path):
        cfg = self._cfg(tmp_path, [sys.executable, "-c", "import sys; sys.stderr.write('bucket not found'); sys.exit(3)"])
        with Store(cfg.db_path) as store:
            index = cli.publish_dashboard(cfg, store)
            (event,) = store.query("SELECT * FROM events")
        assert index.is_file()
        assert event["source"] == "dashboard" and "exit code 3: bucket not found" in event["message"]

    def test_a_command_that_is_not_there_is_logged_too(self, tmp_path):
        cfg = self._cfg(tmp_path, ["/nowhere/uploader", "{dir}"])
        with Store(cfg.db_path) as store:
            cli.publish_dashboard(cfg, store)
            (event,) = store.query("SELECT * FROM events")
        assert "after_dashboard command failed" in event["message"]

    def test_no_command_configured_runs_nothing(self, tmp_path):
        cfg = self._cfg(tmp_path, [])
        with Store(cfg.db_path) as store:
            cli.publish_dashboard(cfg, store)
            assert store.query("SELECT * FROM events") == []
