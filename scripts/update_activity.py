"""Append one explicitly automated check-in per Moscow calendar day."""

from datetime import datetime
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
MOSCOW = ZoneInfo("Europe/Moscow")


def record_run(path, run_url, now=None):
    """Return True when a new daily record is written; preserve existing text."""
    now = datetime.now(MOSCOW) if now is None else now
    if now.tzinfo is None:
        raise ValueError("now must include a timezone")
    day = now.astimezone(MOSCOW).date().isoformat()
    prefix = f"| {day} |"
    content = path.read_text(encoding="utf-8")
    if any(line.startswith(prefix) for line in content.splitlines()):
        return False
    entry = f"{prefix} Автотесты пройдены; запись создана автоматически | [Запуск]({run_url}) |"
    path.write_text(content.rstrip() + "\n" + entry + "\n", encoding="utf-8")
    return True


def main():
    run_url = os.environ.get("ACTIVITY_RUN_URL")
    if not run_url:
        raise SystemExit("Запусти Daily activity через вкладку Actions на GitHub.")
    changed = record_run(ROOT / "activity/README.md", run_url)
    print("Daily check-in added." if changed else "Today's check-in already exists.")


if __name__ == "__main__":
    main()
