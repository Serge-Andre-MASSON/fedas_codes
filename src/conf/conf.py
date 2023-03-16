from os.path import dirname
from pathlib import Path
from yaml import load, Loader

CONF_PATH = Path(f"{dirname(__file__)}") / "conf.yaml"


def get_conf(sub_conf: str = None, conf_path: Path = CONF_PATH) -> dict:
    with open(conf_path) as f:
        conf = load(f, Loader=Loader)

    if sub_conf is None:
        return conf
    return conf[sub_conf]
