import json,pathlib
C=json.loads(pathlib.Path(__file__).with_name("catalog.json").read_text())
IDS=C["cards"];NAMES=C["names"];RELIC_IDS=C["relics"]
