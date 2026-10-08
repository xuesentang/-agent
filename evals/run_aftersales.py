"""Run labeled extraction examples against the configured upstream model."""

import asyncio
import json
from pathlib import Path

from customer_service.aftersales import ExtractionService
from customer_service.config import Settings, build_model


async def main() -> int:
    settings = Settings.from_env()
    service = ExtractionService(build_model(settings), settings.structured_output_method)
    samples = Path(__file__).with_name("aftersales.jsonl")
    failed = 0
    for number, line in enumerate(samples.read_text(encoding="utf-8").splitlines(), 1):
        sample = json.loads(line)
        try:
            actual = (await service.extract(sample["description"])).model_dump()
            match = actual == sample["expected"]
        except Exception as exc:
            actual = {"error": type(exc).__name__}
            match = False
        print(json.dumps({"sample": number, "pass": match, "actual": actual}, ensure_ascii=False))
        failed += not match
    print(f"{number - failed}/{number} samples matched")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
