import json
import math
import re
from pathlib import Path


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class StrategyConfigStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def _validate_id(value: str, label: str):
        if not _ID.fullmatch(value):
            raise ValueError(f"Invalid {label}: {value!r}")

    @classmethod
    def _value(cls, key: str, spec: dict, raw):
        field_type = spec.get("type", "number")
        if field_type == "number_list":
            if not isinstance(raw, list):
                raise ValueError(f"{key} must be a list")
            minimum = int(spec.get("min_items", 0))
            maximum = int(spec.get("max_items", len(raw)))
            if not minimum <= len(raw) <= maximum:
                raise ValueError(f"{key} has an invalid number of items")
            return [cls._number(f"{key}[{index}]", spec, value) for index, value in enumerate(raw)]
        if field_type != "number":
            raise ValueError(f"Unknown config field type: {field_type}")
        return cls._number(key, spec, raw)

    @staticmethod
    def _number(key: str, spec: dict, raw) -> float:
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if value < float(spec.get("min", value)) or value > float(spec.get("max", value)):
            raise ValueError(f"{key} is outside its allowed range")
        return value

    @classmethod
    def _defaults(cls, schema: dict) -> dict[str, object]:
        return {
            key: cls._value(key, spec, spec.get("default", [] if spec.get("type") == "number_list" else 1.0))
            for key, spec in schema.items()
        }

    def _normalize(self, schema: dict, weights: dict, base: dict | None = None) -> dict[str, float]:
        unknown = set(weights) - set(schema)
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
        normalized = dict(base or self._defaults(schema))
        for key, raw in weights.items():
            normalized[key] = self._value(key, schema[key], raw)
        return normalized

    def _path(self, strategy_id: str, config_id: str) -> Path:
        self._validate_id(strategy_id, "strategy id")
        self._validate_id(config_id, "config id")
        return self.root / strategy_id / f"{config_id}.json"

    def get(self, strategy_id: str, version: str, schema: dict, config_id: str) -> dict:
        path = self._path(strategy_id, config_id)
        if config_id == "default" and not path.is_file():
            return {
                "id": "default",
                "strategy_id": strategy_id,
                "strategy_version": version,
                "revision": 0,
                "weights": self._defaults(schema),
            }
        if not path.is_file():
            raise FileNotFoundError(config_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "id": config_id,
            "strategy_id": strategy_id,
            "strategy_version": version,
            "revision": int(payload.get("revision", 1)),
            "weights": self._normalize(schema, payload.get("weights", {})),
        }

    def list(self, strategy_id: str, version: str, schema: dict) -> list[dict]:
        presets = [self.get(strategy_id, version, schema, "default")]
        directory = self.root / strategy_id
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                if path.stem == "default":
                    continue
                presets.append(self.get(strategy_id, version, schema, path.stem))
        return presets

    def save(self, strategy_id: str, version: str, schema: dict, config_id: str, weights: dict) -> dict:
        path = self._path(strategy_id, config_id)
        previous = self.get(strategy_id, version, schema, config_id) if path.is_file() else None
        normalized = self._normalize(
            schema,
            weights,
            previous["weights"] if previous else None,
        )
        payload = {
            "id": config_id,
            "strategy_id": strategy_id,
            "strategy_version": version,
            "revision": (previous["revision"] + 1) if previous else 1,
            "weights": normalized,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return payload

    def delete(self, strategy_id: str, config_id: str):
        if config_id == "default":
            raise ValueError("The default config is read-only")
        path = self._path(strategy_id, config_id)
        if not path.is_file():
            raise FileNotFoundError(config_id)
        path.unlink()
