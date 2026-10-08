# Hermes Weixin integration

The `media_delivery` user plugin provides `/rtsp` and the `deliver_file`
tool. It is tested with Hermes v0.21.5 (release v2026.9.24) and uses
Hermes' built-in Weixin adapter.

## Requirements

- A Hermes gateway with Weixin already configured and logged in.
- `ffmpeg` available on the gateway's PATH.
- A readable RTSP stream, normally `rtsp://localhost:8554/cam` from
  this project's MediaMTX service.

## Install or update

Hermes v0.21.5 only dispatches built-in slash commands while a conversation is
busy. The included generic patch adds `busy_policy="dispatch"` to plugin command
registration and routes plugin commands through both busy-session guards. Other
plugins default to an explicit busy rejection. Authorization and session context
binding still run before the handler.

Apply the patch once to the Hermes checkout used by the gateway:

```bash
hermes_repo=/home/chun/Develop/hermes-agent-v2026.9.24
patch_file="$PWD/integrations/hermes/patches/plugin-busy-dispatch.patch"
git -C "$hermes_repo" apply --check "$patch_file"
git -C "$hermes_repo" apply "$patch_file"
```

After an upgrade, check whether Hermes includes this fix or reapply the patch.
`git -C "$hermes_repo" apply --reverse --check "$patch_file"` succeeds when this
exact patch is already applied. The plugin still loads without the patch, but
logs a warning and cannot run immediately during an active conversation.

From the root of this repository, install the two source files into the
user plugin directory, then restart the gateway:

```bash
plugin_dir="${HERMES_HOME:-$HOME/.hermes}/plugins/media_delivery"
install -d "$plugin_dir"
install -m 644 integrations/hermes/media_delivery/__init__.py integrations/hermes/media_delivery/plugin.yaml "$plugin_dir/"
systemctl --user restart hermes-gateway.service
```

This replaces the installed plugin source. After changing the source in this
repository, repeat these steps to update the running installation.
The gateway uses existing Weixin credentials from its environment or the
account file saved by QR login. Credentials and account files are not included
in this repository.

## Usage

Send `/rtsp` in the Weixin conversation to capture and send one JPEG frame.
The command reads the stream over TCP, allows up to 15 seconds for capture and
30 seconds for image delivery, and reports which stage failed. A delivery timeout
means receipt was not confirmed; it does not prove the image was not delivered.
Duplicate requests in the same chat receive an in-progress notice. Temporary
images and capture processes are cleaned up on completion, timeout, or cancellation.

The plugin logs capture, upload completion, and failures with a request ID and
elapsed time in the gateway logs, without logging RTSP URLs or credentials.

Use `/rtsp rtsp://host:8554/path` for another stream. To change the default,
set `HERMES_RTSP_URL` in the gateway's environment and restart it.

The plugin reads the existing RTSP stream and requires no MediaMTX configuration
changes. The only Hermes core changes are the generic busy-command dispatch patch.

## Verification

The regression tests cover both busy-session guards, authorization, current-chat
routing, duplicate requests, stalled uploads, cancellation, and process cleanup.
They use isolated Hermes homes and never send real Weixin messages:

```bash
PYTHONPATH="$hermes_repo" "$hermes_repo/.venv/bin/python" -m unittest discover -s integrations/hermes/tests -v
```
