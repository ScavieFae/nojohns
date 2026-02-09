# Headless Mode

Running Dolphin without a display — for cloud/CI environments or unattended auto-opponents.

## Current State

Slippi Dolphin requires a display. This is a known limitation (issue #16).

## Known Approaches

- **Xvfb (Linux)**: Virtual framebuffer. Works for basic operation: `xvfb-run nojohns auto phillip --no-wager`
- **Headless Dolphin build**: Vlad (slippi-ai author) has info on building Dolphin without a display requirement. Mattie has details — this is the long-term path.
- **VNC**: Run a VNC server on a cloud box, Dolphin renders into the virtual display. Heavier than Xvfb but lets you visually debug.

## Not Tackling Yet

This is future work. For now:
- macOS: runs normally with display
- Linux with display: runs normally
- Linux headless: use `xvfb-run` as a workaround
- Cloud/CI: not yet supported

When we revisit this, the goal is zero-display operation for cloud deployments.
