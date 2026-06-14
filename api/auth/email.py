"""Email delivery for account flows.

Two backends, selected by ``PHENTRIEVE_EMAIL_BACKEND``:

* ``console`` (default) - logs the message (including the action link) so local
  dev and tests can copy the verification / reset link without an SMTP server.
* ``smtp`` - sends via the standard library :mod:`smtplib`, run in a thread
  pool so it does not block the event loop. No third-party dependency.

The sender address defaults to ``noreply@phentrieve.org``.
"""

from __future__ import annotations

import logging
import smtplib
import ssl
import sys
from email.message import EmailMessage
from typing import Protocol

from fastapi.concurrency import run_in_threadpool

import api.config as api_config

logger = logging.getLogger(__name__)

__all__ = [
    "EmailSender",
    "ConsoleEmailSender",
    "SmtpEmailSender",
    "get_email_sender",
    "build_verify_email",
    "build_reset_email",
]


class EmailSender(Protocol):
    async def send(self, *, to: str, subject: str, text: str) -> None: ...


class ConsoleEmailSender:
    """Writes emails to stdout/logs instead of sending them (local dev / tests).

    Mirrors Django's console email backend: the message (including any action
    link) is written to stdout so it is visible regardless of the surrounding
    log configuration (e.g. under uvicorn, which suppresses app INFO logs). A
    matching ``logger.info`` is also emitted so test harnesses can capture it.
    """

    async def send(self, *, to: str, subject: str, text: str) -> None:
        sender = api_config.PHENTRIEVE_EMAIL_FROM
        rendered = (
            "\n----- [phentrieve console email] -----\n"
            f"From: {sender}\nTo: {to}\nSubject: {subject}\n\n{text}\n"
            "--------------------------------------\n"
        )
        sys.stdout.write(rendered)
        sys.stdout.flush()
        logger.info(
            "[email:console] to=%s from=%s subject=%s\n%s",
            to,
            sender,
            subject,
            text,
        )


class SmtpEmailSender:
    """Sends email through an SMTP server using the standard library."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        tls: str,
        sender: str,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.tls = tls
        self.sender = sender

    def _send_sync(self, *, to: str, subject: str, text: str) -> None:
        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = to
        message["Subject"] = subject
        message.set_content(text)

        if self.tls == "ssl":
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.host, self.port, context=context) as server:
                self._authenticate_and_send(server, message)
        else:
            with smtplib.SMTP(self.host, self.port) as server:
                if self.tls == "starttls":
                    server.starttls(context=ssl.create_default_context())
                self._authenticate_and_send(server, message)

    def _authenticate_and_send(
        self, server: smtplib.SMTP, message: EmailMessage
    ) -> None:
        if self.username and self.password:
            server.login(self.username, self.password)
        server.send_message(message)

    async def send(self, *, to: str, subject: str, text: str) -> None:
        await run_in_threadpool(self._send_sync, to=to, subject=subject, text=text)


def get_email_sender() -> EmailSender:
    """Return the configured email sender backend."""
    if api_config.PHENTRIEVE_EMAIL_BACKEND == "smtp":
        return SmtpEmailSender(
            host=api_config.PHENTRIEVE_SMTP_HOST,
            port=api_config.PHENTRIEVE_SMTP_PORT,
            username=api_config.PHENTRIEVE_SMTP_USERNAME,
            password=api_config.PHENTRIEVE_SMTP_PASSWORD,
            tls=api_config.PHENTRIEVE_SMTP_TLS,
            sender=api_config.PHENTRIEVE_EMAIL_FROM,
        )
    return ConsoleEmailSender()


def build_verify_email(link: str) -> tuple[str, str]:
    """Return (subject, plain-text body) for an email-verification message."""
    subject = "Verify your Phentrieve account"
    text = (
        "Welcome to Phentrieve.\n\n"
        "Please verify your email address to activate your account and unlock "
        "the higher daily full-text quota:\n\n"
        f"{link}\n\n"
        "If you did not create a Phentrieve account, you can ignore this email.\n"
    )
    return subject, text


def build_reset_email(link: str) -> tuple[str, str]:
    """Return (subject, plain-text body) for a password-reset message."""
    subject = "Reset your Phentrieve password"
    text = (
        "We received a request to reset your Phentrieve password.\n\n"
        "Use the link below to choose a new password:\n\n"
        f"{link}\n\n"
        "If you did not request a password reset, you can ignore this email.\n"
    )
    return subject, text
