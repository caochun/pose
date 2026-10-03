"""Media delivery plugin.

Registers a ``deliver_file`` tool so the LLM can send local files to the
user via WeChat by calling a structured tool.

sender_id is captured from pre_llm_call (which receives it from the gateway)
and stored in a module-level variable. The tool reads it when invoked.
"""

from __future__ import annotations

import asyncio
import logging
import json
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_last_sender_id: list[str] = [""]  # updated each turn by pre_llm_call
DEFAULT_RTSP_URL = "rtsp://localhost:8554/cam"


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
        try:
            from gateway.session_context import get_session_env

            platform = get_session_env("HERMES_SESSION_PLATFORM", "").strip().lower()
            chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "").strip()
        except Exception:
            platform = ""
            chat_id = ""

        if platform and platform != "weixin":
            return "RTSP capture is currently supported through Weixin only."

        chat_id = chat_id or _last_sender_id[0] or os.getenv("WEIXIN_HOME_CHANNEL", "").strip()
        if not chat_id:
            return "RTSP capture failed: no Weixin chat target."

        rtsp_url = raw_args.strip() or os.getenv("HERMES_RTSP_URL", DEFAULT_RTSP_URL).strip()
        if not rtsp_url.startswith(("rtsp://", "rtsps://")):
            return "RTSP capture failed: URL must start with rtsp:// or rtsps://."

        fd, frame_path = tempfile.mkstemp(suffix=".jpg", prefix="rtsp_frame_")
        os.close(fd)
        try:
            try:
                process = await asyncio.create_subprocess_exec(
                    "ffmpeg",
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
                    "-loglevel",
                    "error",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                return "RTSP capture failed: ffmpeg is not installed."

            try:
                _, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return "RTSP capture failed: timed out after 15s."

            if process.returncode != 0:
                lines = stderr.decode("utf-8", errors="replace").strip().splitlines()
                detail = (lines[-1] if lines else "unknown error")[:200]
                return f"RTSP capture failed: {detail}"

            if not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
                return "RTSP capture failed: empty frame."

            from gateway.platforms.weixin import send_weixin_direct

            extra, token = _weixin_credentials()
            result = await send_weixin_direct(
                extra=extra,
                token=token,
                chat_id=chat_id,
                message="",
                media_files=[(frame_path, False)],
            )
            if result.get("success"):
                return ""
            return f"RTSP capture succeeded but Weixin send failed: {result.get('error')}"
        except Exception as exc:
            logger.exception("rtsp capture failed")
            return f"RTSP capture failed: {exc}"
        finally:
            try:
                os.unlink(frame_path)
            except OSError:
                pass

    ctx.register_tool(
        name="deliver_file",
        toolset="media_delivery",
        schema=DELIVER_SCHEMA,
        handler=handle_deliver_file,
        is_async=True,
    )
    ctx.register_command(
        "rtsp",
        handler=handle_rtsp,
        description="Capture one frame from the local RTSP camera and send it via Weixin.",
        args_hint="[RTSP_URL]",
    )
