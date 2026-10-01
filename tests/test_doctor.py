import json

from app.doctor import _shard_check


def test_shard_check_requires_all_files(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "groups_total": 2,
        "shards": {
            "a": {"file": "a.safetensors"},
            "b": {"file": "b.safetensors"},
        },
    }))
    (tmp_path / "a.safetensors").write_bytes(b"a")
    assert _shard_check("model", tmp_path).status == "pending"
    (tmp_path / "b.safetensors").write_bytes(b"b")
    assert _shard_check("model", tmp_path).status == "ok"
