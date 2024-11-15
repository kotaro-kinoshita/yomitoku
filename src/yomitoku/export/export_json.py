import json


def export_json(inputs, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            inputs.dict(),
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
