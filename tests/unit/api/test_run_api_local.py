from pathlib import Path

from api import run_api_local


def test_setup_environment_loads_local_config_after_repo_env_with_override(
    mocker, monkeypatch, tmp_path
) -> None:
    api_dir = tmp_path / "api"
    api_dir.mkdir()
    config_path = api_dir / "local_api_config.env"
    config_path.write_text("PHENTRIEVE_DATA_ROOT_DIR=./data\n")
    project_env_path = tmp_path / ".env"
    project_env_path.write_text("PHENTRIEVE_DATA_ROOT_DIR=./root-data\n")

    mocker.patch.object(run_api_local, "__file__", str(api_dir / "run_api_local.py"))
    load_dotenv = mocker.patch("api.run_api_local.load_dotenv")
    mocker.patch("api.run_api_local.ensure_directory_exists", side_effect=Path)
    monkeypatch.setenv("PHENTRIEVE_DATA_ROOT_DIR", str(tmp_path / "resolved-data"))
    monkeypatch.setenv("PHENTRIEVE_DATA_DIR", str(tmp_path / "resolved-data"))
    monkeypatch.setenv("PHENTRIEVE_INDEX_DIR", str(tmp_path / "resolved-indexes"))
    monkeypatch.setenv("PHENTRIEVE_RESULTS_DIR", str(tmp_path / "resolved-results"))

    run_api_local.setup_environment()

    assert load_dotenv.call_args_list == [
        mocker.call(project_env_path, override=False),
        mocker.call(config_path, override=True),
    ]
