import hashlib
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo
import requests

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def features_checksum(features: list[str]) -> str:
    return sha256_text("\n".join(features))

def send_pushover(title: str, message: str, token: str, user: str) -> None:
    if not token or not user:
        return

    try:
        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": token,
                "user": user,
                "title": title,
                "message": message,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"Pushover error {resp.status_code}: {resp.text}")
    except Exception as exc:  # pragma: no cover - best-effort notification
        print(f"Pushover exception: {exc}")


def is_regular_trading_time(now: datetime | None = None) -> bool:
    """Return True if the current time is within US equity RTH (9:30-16:00 ET)."""

    eastern = ZoneInfo("America/New_York")
    now = now or datetime.now(tz=eastern)

    if now.tzinfo is None:
        now = now.replace(tzinfo=eastern)

    if now.weekday() > 4:
        return False

    start = time(9, 30, tzinfo=eastern)
    end = time(16, 0, tzinfo=eastern)
    return start <= now.timetz() < end
