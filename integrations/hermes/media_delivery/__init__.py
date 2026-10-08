"""Media delivery plugin.

Registers a ``deliver_file`` tool so the LLM can send local files to the
user via WeChat by calling a structured tool.

sender_id is captured from pre_llm_call (which receives it from the gateway)
and stored in a module-level variable. The tool reads it when invoked.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import json
import os
import tempfile
import time
import uuid
from contextlib import suppress
from pathlib import Path

logger = logging.getLogger(__name__)

_last_sender_id: list[str] = [""]  # updated each turn by pre_llm_call
DEFAULT_RTSP_URL = "rtsp://localhost:8554/cam"
CAPTURE_TIMEOUT = 15
SEND_TIMEOUT = 30


def _weixin_credentials() -> tuple[dict[str, str], str | None]:
    """Load documented Weixin settings, including credentials saved by QR login."""
    account_id = os.getenv("WEIXIN_ACCOUNT_ID", "").strip()
    token = os.getenv("WEIXIN_TOKEN", "").strip()
    base_url = os.getenv("WEIXIN_BASE_URL", "").strip()
    cdn_base_url = os.getenv("WEIXIN_CDN_BASE_URL", "").strip()

    if account_id and not token:
        hermes_home = Path(os.getenv("HERMES_HOME", "~/.hermes")).expanduser()
        account_file = hermes_home / "weixin" / "accounts" / f"{account_id}.json"
        try:
            saved = json.loads(account_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved = {}
        token = str(saved.get("token") or "").strip()
        base_url = base_url or str(saved.get("base_url") or "").strip()

    extra = {"account_id": account_id}
    if base_url:
        extra["base_url"] = base_url
    if cdn_base_url:
        extra["cdn_base_url"] = cdn_base_url
    return extra, token or None


DELIVER_SCHEMA = {
    "name": "deliver_file",
    "description": (
        "REQUIRED: Send a local file (image, video, or document) to the user via WeChat. "
        "You MUST call this tool whenever the user asks you to send a file or image. "
        "Simply describing or mentioning a file does NOT send it — "
        "only calling this tool actually delivers it to the user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to send.",
            },
        },
        "required": ["file_path"],
    },
}


def register(ctx) -> None:
    rtsp_in_flight: set[str] = set()

    def pre_llm_call(sender_id: str = "", **kwargs):
        if sender_id:
            _last_sender_id[0] = sender_id
        return None

    ctx.register_hook("pre_llm_call", pre_llm_call)

    async def handle_deliver_file(file_path, **kwargs) -> str:
        # LLM sometimes passes file_path as a dict e.g. {"path": "/foo"} or {"file_path": "/foo"}
        if isinstance(file_path, dict):
            file_path = (
                file_path.get("path")
                or file_path.get("file_path")
                or file_path.get("value")
                or next(iter(file_path.values()), "")
            )
        path = Path(str(file_path)).expanduser()
        if not path.is_file():
            return f"Error: file not found: {file_path}"

        chat_id = _last_sender_id[0] or os.getenv("WEIXIN_HOME_CHANNEL", "")
        if not chat_id:
            return "Error: could not determine recipient"

        try:
            from gateway.platforms.weixin import send_weixin_direct
            extra, token = _weixin_credentials()
            result = await send_weixin_direct(
                extra=extra,
                token=token,
                chat_id=chat_id,
                message="",
                media_files=[(str(path), False)],
            )
            if result.get("success"):
                return f"Sent {path.name} to user."
            return f"Send failed: {result.get('error')}"
        except Exception as exc:
            logger.exception("deliver_file failed")
            return f"Error: {exc}"

    async def handle_rtsp(raw_args: str = "") -> str:
        """Capture one RTSP frame and send it to the current Weixin chat."""
        from gateway.session_context import get_session_env

        platform = get_session_env("HERMES_SESSION_PLATFORM", "").strip().lower()
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "").strip()
        if platform != "weixin":
            return "RTSP capture is currently supported through Weixin only."
        if not chat_id:
            return "RTSP capture failed: no Weixin chat target."
        if chat_id in rtsp_in_flight:
            return "An RTSP snapshot is already being captured or sent. Please wait for its result."

        rtsp_url = raw_args.strip() or os.getenv("HERMES_RTSP_URL", DEFAULT_RTSP_URL).strip()
        if not rtsp_url.startswith(("rtsp://", "rtsps://")):
            return "RTSP capture failed: URL must start with rtsp:// or rtsps://."

        request_id = uuid.uuid4().hex[:8]
        started = time.monotonic()
        process = None
        frame_path = None
        stage = "capture"
        rtsp_in_flight.add(chat_id)
        logger.info("rtsp[%s] capture started", request_id)
        try:
            fd, frame_path = tempfile.mkstemp(suffix=".jpg", prefix="rtsp_frame_")
            os.close(fd)
            try:
                process = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-rtsp_transport",
                    "tcp",
                    "-i",
                    rtsp_url,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    frame_path,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                logger.warning("rtsp[%s] ffmpeg not installed", request_id)
                return "RTSP capture failed: ffmpeg is not installed."

            try:
                _, stderr = await asyncio.wait_for(process.communicate(), timeout=CAPTURE_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("rtsp[%s] capture timed out after %ss", request_id, CAPTURE_TIMEOUT)
                return f"RTSP capture timed out after {CAPTURE_TIMEOUT}s. Check the camera stream."

            if process.returncode != 0:
                lines = stderr.decode("utf-8", errors="replace").strip().splitlines()
                detail = (lines[-1] if lines else "unknown error").replace(rtsp_url, "[RTSP stream]")[:200]
                logger.warning("rtsp[%s] ffmpeg exited with code %s", request_id, process.returncode)
                return f"RTSP capture failed: {detail}"

            if not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
                logger.warning("rtsp[%s] capture produced an empty frame", request_id)
                return "RTSP capture failed: empty frame."

            from gateway.platforms.weixin import send_weixin_direct

            extra, token = _weixin_credentials()
            stage = "send"
            logger.info("rtsp[%s] captured %s bytes in %.1fs; sending image", request_id,
                        os.path.getsize(frame_path), time.monotonic() - started)
            try:
                result = await asyncio.wait_for(
                    send_weixin_direct(
                        extra=extra, token=token, chat_id=chat_id, message="",
                        media_files=[(frame_path, False)],
                    ),
                    timeout=SEND_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("rtsp[%s] Weixin send timed out after %ss", request_id, SEND_TIMEOUT)
                return (f"RTSP frame captured, but Weixin delivery was not confirmed within {SEND_TIMEOUT}s. "
                        "If no image arrives, try /rtsp again.")
            if result.get("success"):
                logger.info("rtsp[%s] image sent in %.1fs", request_id, time.monotonic() - started)
                return ""
            logger.warning("rtsp[%s] Weixin rejected image delivery", request_id)
            return f"RTSP capture succeeded but Weixin send failed: {result.get('error')}"
        except asyncio.CancelledError:
            logger.info("rtsp[%s] cancelled during %s", request_id, stage)
            raise
        except Exception as exc:
            logger.warning("rtsp[%s] %s failed: %s", request_id, stage, type(exc).__name__)
            return f"RTSP {stage} failed: {type(exc).__name__}. Please try again."
        finally:
            try:
                if process is not None and process.returncode is None:
                    with suppress(ProcessLookupError):
                        process.kill()
                    await process.communicate()
            finally:
                if frame_path:
                    with suppress(OSError):
                        os.unlink(frame_path)
                rtsp_in_flight.discard(chat_id)

    ctx.register_tool(
        name="deliver_file",
        toolset="media_delivery",
        schema=DELIVER_SCHEMA,
        handler=handle_deliver_file,
        is_async=True,
    )
    command_options = {}
    if "busy_policy" in inspect.signature(ctx.register_command).parameters:
        command_options["busy_policy"] = "dispatch"
    else:
        logger.warning("rtsp: Hermes needs the plugin busy-dispatch patch to handle commands during agent turns")
    ctx.register_command(
        "rtsp",
        handler=handle_rtsp,
        description="Capture one frame from the local RTSP camera and send it via Weixin.",
        args_hint="[RTSP_URL]",
        **command_options,
    )
