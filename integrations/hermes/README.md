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
The command reads the stream over TCP, allows up to 15 seconds for capture,
and deletes the temporary image after the send attempt.

Use `/rtsp rtsp://host:8554/path` for another stream. To change the default,
set `HERMES_RTSP_URL` in the gateway's environment and restart it.

The plugin reads the existing RTSP stream; installation requires no changes
to MediaMTX or Hermes core files.
