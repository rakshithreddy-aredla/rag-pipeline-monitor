"""Pluggable alert sinks (TODO Phase 12).

The dispatcher fires AT MOST ONCE per run_id — repeated CUSUM checks on the
same run must never spam sinks (Phase 12 test requirement).
"""

from __future__ import annotations

from typing import Any, Protocol

from core.engine import Alert


class AlertSink(Protocol):
    def send(self, alert: Alert) -> None: ...


class LogSink:
    """Always-available sink; structured log line."""

    def __init__(self) -> None:
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> None:
        print(
            f"[CASCADEGUARD ALERT] run={alert.run_id} step={alert.triggered_at_step} "
            f"S_n={alert.s_n_value:.4f} root_causes={[c.step_id for c in alert.root_causes]}"
        )
        self.sent.append(alert)


class WebhookSink:
    """Generic webhook (httpx). Configure URL in configs alerts.sinks."""

    def __init__(self, url: str, timeout_s: float = 5.0) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> None:
        import httpx

        payload = {
            "event": "cascadeguard.alert",
            "run_id": alert.run_id,
            "triggered_at_step": alert.triggered_at_step,
            "s_n_value": alert.s_n_value,
            "root_causes": [
                {"step_id": rc.step_id, "share": rc.share, "path_length": rc.path_length}
                for rc in alert.root_causes
            ],
        }
        httpx.post(self.url, json=payload, timeout=self.timeout_s)
        self.sent.append(alert)


class SlackSink(WebhookSink):
    """Slack Incoming Webhook formatting."""

    def send(self, alert: Alert) -> None:
        import httpx

        causes = ", ".join(f"`{rc.step_id}` ({rc.share:.0%})" for rc in alert.root_causes)
        text = (
            f":rotating_light: CascadeGuard alert — run `{alert.run_id}` tripped at "
            f"step `{alert.triggered_at_step}` (S_n={alert.s_n_value:.3f})"
            + (f"; likely origin: {causes}" if causes else "")
        )
        httpx.post(self.url, json={"text": text}, timeout=self.timeout_s)
        self.sent.append(alert)


def build_sink(spec: dict[str, Any]) -> AlertSink:
    kind = spec.get("type")
    if kind == "log":
        return LogSink()
    if kind == "webhook":
        return WebhookSink(str(spec["url"]))
    if kind == "slack":
        return SlackSink(str(spec["url"]))
    raise ValueError(f"unknown alert sink type {kind!r}")


class AlertDispatcher:
    """Routes alerts to configured sinks; dedupes per run (fires once per run)."""

    def __init__(self, sinks: list[AlertSink]) -> None:
        self.sinks = sinks
        self._fired_runs: set[str] = set()

    def dispatch(self, result_alerts: list[Alert]) -> int:
        count = 0
        for alert in result_alerts:
            if alert.run_id in self._fired_runs:
                continue
            self._fired_runs.add(alert.run_id)
            for sink in self.sinks:
                sink.send(alert)
            count += 1
        return count
