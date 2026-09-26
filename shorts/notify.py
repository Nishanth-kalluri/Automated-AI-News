"""Stage 11: tell a human what happened, from the pipeline's own AgentMail inbox."""
from __future__ import annotations

import logging
from typing import Protocol

import requests

from .config import Config

log = logging.getLogger(__name__)
AGENTMAIL_API = "https://api.agentmail.to/v0"


class Notifier(Protocol):
    name: str

    def send(self, subject: str, text: str) -> None: ...


class LogNotifier:
    name = "log"

    def send(self, subject: str, text: str) -> None:
        log.info("notify: %s\n%s", subject, text)


class AgentMailNotifier:
    name = "agentmail"

    def __init__(self, api_key: str, inbox: str, to: str):
        self.api_key, self.inbox, self.to = api_key, inbox, to

    def send(self, subject: str, text: str) -> None:
        resp = requests.post(
            f"{AGENTMAIL_API}/inboxes/{self.inbox}/messages/send",
            json={"to": [self.to], "subject": subject, "text": text},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        resp.raise_for_status()


def build_notifier(cfg: Config) -> Notifier:
    if cfg.notify_email and cfg.agentmail_api_key and cfg.agentmail_inbox:
        return AgentMailNotifier(cfg.agentmail_api_key, cfg.agentmail_inbox, cfg.notify_email)
    return LogNotifier()


def notify(notifier: Notifier, subject: str, text: str) -> None:
    try:
        notifier.send(subject, text)
    except Exception as exc:  # a failed email must not fail the run
        log.warning("%s notification failed: %s", notifier.name, exc)
        LogNotifier().send(subject, text)
