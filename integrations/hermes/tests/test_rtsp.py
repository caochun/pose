"""Run with the Hermes source on PYTHONPATH; no real Weixin messages are sent."""

import asyncio
import importlib.util
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.platforms.weixin import WeixinAdapter
from gateway.run_inbound import GatewayInboundMixin
from gateway.session import SessionSource, build_session_key
from gateway.session_context import clear_session_vars, get_session_env, set_session_vars
from hermes_cli.commands import should_bypass_active_session
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def load_plugin():
    path = Path(__file__).resolve().parents[1] / "media_delivery" / "__init__.py"
    spec = importlib.util.spec_from_file_location("rtsp_test_plugin", path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    return plugin


class RtspTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict("os.environ", {"HERMES_HOME": self.temp.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.plugin = load_plugin()
        self.ctx = SimpleNamespace(register_hook=lambda *a, **k: None,
                                   register_tool=lambda **k: None)

        def register_command(name, handler, busy_policy="reject", **kwargs):
            self.handler = handler
            self.policy = busy_policy

        self.ctx.register_command = register_command
        self.plugin.register(self.ctx)
        self.paths = []

        async def capture(*args, **kwargs):
            self.paths.append(Path(args[-1]))
            self.paths[-1].write_bytes(b"jpeg-frame")
            return SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b"", b"")))

        self.capture = patch.object(self.plugin.asyncio, "create_subprocess_exec", side_effect=capture)
        self.capture_mock = self.capture.start()
        self.addCleanup(self.capture.stop)
        self.send = patch("gateway.platforms.weixin.send_weixin_direct", new_callable=AsyncMock)
        self.sender = self.send.start()
        self.sender.return_value = {"success": True}
        self.addCleanup(self.send.stop)
        self.tokens = set_session_vars(platform="weixin", chat_id="test-chat")
        self.addCleanup(clear_session_vars, self.tokens)

    async def test_success_routes_to_current_chat_and_cleans_image(self):
        self.plugin._last_sender_id[0] = "stale-chat"
        self.assertEqual(await self.handler(), "")
        self.assertEqual(self.sender.call_args.kwargs["chat_id"], "test-chat")
        self.assertTrue(self.paths)
        self.assertTrue(all(not path.exists() for path in self.paths))
        self.assertEqual(self.policy, "dispatch")

    async def test_missing_context_never_uses_stale_recipient(self):
        clear_session_vars(self.tokens)
        self.plugin._last_sender_id[0] = "stale-chat"
        self.assertIn("Weixin only", await self.handler())
        self.capture_mock.assert_not_called()
        self.sender.assert_not_awaited()

    async def test_send_timeout_cancels_upload_and_allows_next_request(self):
        cancelled = asyncio.Event()

        async def stuck_send(**kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.plugin.SEND_TIMEOUT = 0.01
        self.sender.side_effect = stuck_send
        result = await self.handler()
        self.assertIn("delivery was not confirmed", result)
        self.assertTrue(cancelled.is_set())
        self.assertFalse(self.paths[-1].exists())
        self.sender.side_effect = None
        self.assertEqual(await self.handler(), "")

    async def test_duplicate_command_does_not_start_second_capture(self):
        sending = asyncio.Event()
        release = asyncio.Event()

        async def delayed_send(**kwargs):
            sending.set()
            await release.wait()
            return {"success": True}

        self.sender.side_effect = delayed_send
        first = asyncio.create_task(self.handler())
        try:
            await asyncio.wait_for(sending.wait(), 1)
            self.assertIn("already", await self.handler())
            self.assertEqual(self.capture_mock.call_count, 1)
        finally:
            release.set()
            await first

    async def test_failed_delivery_is_reported_and_image_removed(self):
        self.sender.return_value = {"error": "session not ready"}
        self.assertIn("session not ready", await self.handler())
        self.assertFalse(self.paths[-1].exists())

    async def test_capture_timeout_and_cancellation_reap_real_subprocess(self):
        self.capture.stop()
        create_process = asyncio.create_subprocess_exec
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                started = asyncio.Event()
                processes = []

                async def stalled_capture(*args, **kwargs):
                    self.paths.append(Path(args[-1]))
                    process = await create_process(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
                    processes.append(process)
                    started.set()
                    return process

                self.plugin.CAPTURE_TIMEOUT = 30 if cancel else 0.05
                with patch.object(self.plugin.asyncio, "create_subprocess_exec", side_effect=stalled_capture):
                    task = asyncio.create_task(self.handler())
                    await asyncio.wait_for(started.wait(), 2)
                    if cancel:
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        self.assertIn("capture timed out", await asyncio.wait_for(task, 2))
                self.assertIsNotNone(processes[0].returncode)
                self.assertFalse(self.paths[-1].exists())
        self.sender.assert_not_awaited()


class BusyRunner(GatewayInboundMixin):
    def __init__(self):
        self.config = GatewayConfig()
        self._draining = False
        self.denied = None

    def _check_slash_access(self, source, command):
        return self.denied

    def _session_key_for_source(self, source):
        return build_session_key(source)

    @contextmanager
    def _session_env_scope(self, context):
        tokens = set_session_vars(platform=context.source.platform.value,
                                  chat_id=context.source.chat_id)
        try:
            yield
        finally:
            clear_session_vars(tokens)


class BusyDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        env = patch.dict("os.environ", {"HERMES_HOME": self.temp.name})
        env.start()
        self.addCleanup(env.stop)
        self.manager = PluginManager()
        self.ctx = PluginContext(PluginManifest(name="test", source="user"), self.manager)
        discovery = patch("hermes_cli.plugins._ensure_plugins_discovered", return_value=self.manager)
        discovery.start()
        self.addCleanup(discovery.stop)
        self.runner = BusyRunner()
        self.event = MessageEvent(text="/rtsp", source=SessionSource(
            platform=Platform.WEIXIN, chat_id="test-chat", user_id="test-user"))
        self.handler = AsyncMock(return_value="sent")

    async def dispatch(self):
        return await self.runner._hm_busy_slash_or_photo(self.event, self.event.source, "busy-session")

    async def test_adapter_and_runner_dispatch_without_queue_or_agent(self):
        async def handler(args):
            self.assertEqual(get_session_env("HERMES_SESSION_CHAT_ID"), "test-chat")
            return "snapshot sent"

        self.ctx.register_command("rtsp", handler, busy_policy="dispatch")
        self.assertTrue(should_bypass_active_session("rtsp"))
        self.assertFalse(should_bypass_active_session("missing-command"))
        adapter = WeixinAdapter(PlatformConfig(enabled=True))
        adapter._canonicalize = lambda source: None
        replies = []

        async def dispatch_inline(event):
            replies.append(await self.dispatch())

        adapter._dispatch_inline_reply = dispatch_inline
        adapter._busy_session_handler = AsyncMock(side_effect=AssertionError("command queued"))
        await adapter._handle_message_while_active(self.event, "busy-session")
        self.assertEqual(replies, [(True, "snapshot sent")])
        self.assertEqual(adapter._pending_messages, {})

    async def test_default_busy_policy_rejects_explicitly(self):
        self.ctx.register_command("rtsp", self.handler)
        self.assertTrue(should_bypass_active_session("rtsp"))
        handled, response = await self.dispatch()
        self.assertTrue(handled)
        self.assertIn("cannot run mid-turn", response)
        self.handler.assert_not_awaited()

    async def test_denied_command_does_not_capture(self):
        self.ctx.register_command("rtsp", self.handler, busy_policy="dispatch")
        self.runner.denied = "Access denied"
        self.assertEqual(await self.dispatch(), (True, "Access denied"))
        self.handler.assert_not_awaited()

    async def test_noncontrol_text_does_not_invoke_command(self):
        self.ctx.register_command("rtsp", self.handler, busy_policy="dispatch")
        self.event.allow_gateway_control = False
        self.assertEqual(await self.dispatch(), (False, None))
        self.handler.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
