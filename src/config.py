from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path
