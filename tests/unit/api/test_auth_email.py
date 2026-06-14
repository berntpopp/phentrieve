"""Tests for the email senders and message builders."""

import asyncio
import logging

import api.config as api_config
from api.auth.email import (
    ConsoleEmailSender,
    SmtpEmailSender,
    build_reset_email,
    build_verify_email,
    get_email_sender,
)


def test_console_sender_logs_recipient_and_body(caplog):
    caplog.set_level(logging.INFO, logger="api.auth.email")
    sender = ConsoleEmailSender()
    asyncio.run(
        sender.send(to="a@ex.com", subject="Hi", text="link http://x/verify?token=z")
    )
    assert "a@ex.com" in caplog.text
    assert "verify?token=z" in caplog.text


def test_build_verify_email_contains_link():
    subject, text = build_verify_email("https://phentrieve.org/verify?token=abc")
    assert subject
    assert "verify?token=abc" in text


def test_build_reset_email_contains_link():
    subject, text = build_reset_email("https://phentrieve.org/reset-password?token=xy")
    assert subject
    assert "reset-password?token=xy" in text


def test_factory_returns_console_by_default(monkeypatch):
    monkeypatch.setattr(api_config, "PHENTRIEVE_EMAIL_BACKEND", "console")
    assert isinstance(get_email_sender(), ConsoleEmailSender)


def test_factory_returns_smtp_when_configured(monkeypatch):
    monkeypatch.setattr(api_config, "PHENTRIEVE_EMAIL_BACKEND", "smtp")
    assert isinstance(get_email_sender(), SmtpEmailSender)


def test_smtp_sender_builds_message(monkeypatch):
    captured = {}

    class _FakeSMTP:
        def __init__(self, host, port):
            captured["host"] = host
            captured["port"] = port

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self, context=None):
            captured["starttls"] = True

        def login(self, user, pwd):
            captured["login"] = (user, pwd)

        def send_message(self, message):
            captured["from"] = message["From"]
            captured["to"] = message["To"]
            captured["subject"] = message["Subject"]

    monkeypatch.setattr("api.auth.email.smtplib.SMTP", _FakeSMTP)
    sender = SmtpEmailSender(
        host="mail.example.org",
        port=587,
        username="u",
        password="p",
        tls="starttls",
        sender="noreply@phentrieve.org",
    )
    asyncio.run(sender.send(to="a@ex.com", subject="Hi", text="body"))
    assert captured["host"] == "mail.example.org"
    assert captured["starttls"] is True
    assert captured["login"] == ("u", "p")
    assert captured["from"] == "noreply@phentrieve.org"
    assert captured["to"] == "a@ex.com"
