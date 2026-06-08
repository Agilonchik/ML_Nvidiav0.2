import yaml
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    root_dir = Path(__file__).resolve().parents[2]
    full_config_path = root_dir / config_path
    if not full_config_path.exists():
        raise FileNotFoundError(f"Config not found: {full_config_path}")
    with open(full_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    paths = config.get("paths", {})
    for key, folder in paths.items():
        if key != "data_dir":
            dir_path = root_dir / folder
            dir_path.mkdir(parents=True, exist_ok=True)
            config["paths"][key] = str(dir_path)
    config["paths"]["data_dir"] = str(root_dir / paths.get("data_dir", "data"))
    config["root_dir"] = str(root_dir)
    return config
