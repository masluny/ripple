import json
import sys

from fastapi.testclient import TestClient

from ripple import config
from ripple.api import main


def _fake_demo(root):
    d = root / "synthetic" / "demo"
    d.mkdir(parents=True)
    day = {
        "day": "2025-12-01", "model": "lgbm", "groups": ["rotation", "connection", "weather", "airport"],
        "flights": [
            {"id": 1, "no": "LO100", "tail": "SP-LAA", "type": "E195", "dep": "CDG", "arr": "WAW",
             "sd": 300, "sa": 420, "ad": 340, "aa": 455, "prev": None, "truth": "primary"},
            {"id": 2, "no": "LO101", "tail": "SP-LAB", "type": "E195", "dep": "WAW", "arr": "VNO",
             "sd": 480, "sa": 540, "ad": None, "aa": None, "prev": None, "truth": None},
        ],
        "frames": [{"t": 360, "conns": [[1, 2, 60, 0.42]],
                    "preds": [[2, 0, 5, 12, 25, 60, -5, 2, 10, 22, 55, "rotation", [9, 0, 1, 2],
                               "Inherited from the previous leg"]]}],
        "pred_cols": ["id", "d05", "d25", "d50", "d75", "d95", "a05", "a25", "a50", "a75", "a95",
                      "cause", "deltas", "text"],
    }  # fmt: skip
    (d / "2025-12-01_lgbm.json").write_text(json.dumps(day))
    (d / "index.json").write_text(json.dumps({"dataset": "synthetic", "days": ["2025-12-01"],
                                              "models": ["lgbm"], "airports": []}))  # fmt: skip


def test_api_serves_replay_and_template_briefing(tmp_path, monkeypatch):
    _fake_demo(tmp_path)
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    main._index.cache_clear()
    main._day.cache_clear()
    c = TestClient(main.app)
    assert c.get("/api/index").json()["days"] == ["2025-12-01"]
    assert c.get("/api/day/2025-12-01/lgbm").json()["flights"][0]["no"] == "LO100"
    assert c.get("/api/day/2025-12-02/lgbm").status_code == 404
    # force the template path so the test never calls the network
    monkeypatch.setitem(sys.modules, "anthropic", None)
    r = c.post("/api/briefing", json={"day": "2025-12-01", "model": "lgbm", "frame": 0}).json()
    assert r["source"] == "template"
    assert "Synthetic replay" in r["text"] and "LO100" in r["text"]
