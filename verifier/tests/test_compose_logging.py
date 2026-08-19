from pathlib import Path

import yaml

_COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.yml"


def test_worker_uses_json_file_logging_driver():
    compose = yaml.safe_load(_COMPOSE.read_text())
    logging = compose["services"]["worker"]["logging"]
    assert logging["driver"] == "json-file"
    assert logging["options"]["max-size"] == "10m"
    assert logging["options"]["max-file"] == "3"
