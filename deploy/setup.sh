#!/bin/bash
# Setup script for The Clicker (the-clicker), a live TV ad blocker.
#
# Works on any Debian family system with systemd: a Raspberry Pi (Raspberry
# Pi OS), a Debian/Ubuntu machine or mini PC, or a Proxmox LXC container.
#
# Prerequisites (all setups):
#   - Plug the USB capture card into a USB 3.0 port and confirm it appears:
#       ls /dev/video*        # expect /dev/video0 (or similar)
#   - The machine must be on the same network as the device you're controlling
#     (e.g. the Apple TV), so pyatv can reach it.
#   - Run this script as root (or with sudo).
#
# Extra prerequisites for a Proxmox LXC container ONLY (skip on a Pi/bare metal):
#   1. Create a Debian 12 (Bookworm) LXC container.
#   2. Pass the capture card through, on the HOST, find it (ls -la /dev/video*)
#      and add to /etc/pve/lxc/<CTID>.conf:
#        lxc.cgroup2.devices.allow: c 81:* rwm
#        lxc.mount.entry: /dev/video0 dev/video0 none bind,optional,create=file
#   3. Make the node openable from an UNPRIVILEGED container. On the HOST,
#      container root maps to an unprivileged UID, so the default 0660 root:video
#      node (shown as nobody:nogroup inside) gives "Permission denied". Add a udev
#      rule on the host so it's world openable and survives reboots/replugs:
#        echo 'KERNEL=="video[0-9]*", SUBSYSTEM=="video4linux", MODE="0666"' \
#          > /etc/udev/rules.d/99-capture-card.rules
#        udevadm control --reload && udevadm trigger
#      (On a Pi / bare metal you don't need this. The root run service opens the
#       device directly. To run manually as a non root user, add yourself to the
#       `video` group instead: sudo usermod -aG video $USER, then relogin.)

set -e

echo "=== The Clicker Setup ==="

# System packages
echo "[1/5] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3 python3-pip python3-venv \
    python3-opencv \
    v4l-utils \
    git curl

# Python venv
echo "[2/5] Creating Python environment..."
INSTALL_DIR="/opt/the-clicker"
mkdir -p "$INSTALL_DIR"

if [ ! -d "$INSTALL_DIR/venv" ]; then
    python3 -m venv "$INSTALL_DIR/venv"
fi
source "$INSTALL_DIR/venv/bin/activate"

# Python dependencies
echo "[3/5] Installing Python dependencies..."
pip install --quiet \
    pyatv \
    fastapi \
    uvicorn \
    opencv-python-headless \
    numpy

# Copy project files (skip if already in the install directory)
echo "[4/5] Setting up project files..."
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ "$SCRIPT_DIR" != "$INSTALL_DIR" ]; then
    for f in server.py engine.py capture.py detector.py calibrate.py config.py intervention.py pair.py dashboard.html; do
        if [ -f "$SCRIPT_DIR/$f" ]; then
            cp "$SCRIPT_DIR/$f" "$INSTALL_DIR/"
        fi
    done
    # Copy credentials if they exist
    if [ -f "$SCRIPT_DIR/credentials.json" ]; then
        cp "$SCRIPT_DIR/credentials.json" "$INSTALL_DIR/"
        echo "  Copied Apple TV credentials."
    fi
    # Copy any existing profiles
    if [ -d "$SCRIPT_DIR/profiles" ] && ls "$SCRIPT_DIR/profiles"/*.json &>/dev/null; then
        mkdir -p "$INSTALL_DIR/profiles"
        cp "$SCRIPT_DIR/profiles"/* "$INSTALL_DIR/profiles/"
        echo "  Copied existing profiles."
    fi
else
    echo "  Already in install directory, skipping copy."
fi
mkdir -p "$INSTALL_DIR/profiles"

if [ ! -f "$INSTALL_DIR/credentials.json" ]; then
    echo "  No credentials.json, you'll need to pair with the Apple TV."
    echo "  Run: cd $INSTALL_DIR && source venv/bin/activate && python3 pair.py"
fi

# Systemd service
echo "[5/5] Installing systemd service..."
cat > /etc/systemd/system/the-clicker.service << 'EOF'
[Unit]
Description=The Clicker: Live TV Ad Blocker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/the-clicker
ExecStart=/opt/the-clicker/venv/bin/python3 server.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable the-clicker

# Capture device sanity check.
echo ""
echo "Checking capture device access..."
if [ ! -e /dev/video0 ]; then
    echo "  WARNING: /dev/video0 not found."
    echo "  - Pi / bare metal: plug the capture card into a USB 3.0 port and check 'lsusb' / 'v4l2-ctl --list-devices'."
    echo "  - Proxmox LXC: configure USB passthrough in /etc/pve/lxc/<CTID>.conf on the host (see this script's header)."
elif v4l2-ctl -d /dev/video0 --info >/dev/null 2>&1; then
    echo "  OK: /dev/video0 is accessible."
else
    echo "  WARNING: /dev/video0 exists but can't be opened (permission denied)."
    echo "  - Pi / bare metal: run the service as root (the default here), or add your user to the 'video' group."
    echo "  - Proxmox unprivileged LXC: container root can't open the host node. Fix on the PROXMOX HOST:"
    echo "      echo 'KERNEL==\"video[0-9]*\", SUBSYSTEM==\"video4linux\", MODE=\"0666\"' \\"
    echo "        > /etc/udev/rules.d/99-capture-card.rules"
    echo "      udevadm control --reload && udevadm trigger"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Verify capture card: v4l2-ctl --list-devices"
echo "  2. Pair Apple TV (if no credentials copied):"
echo "     cd $INSTALL_DIR && source venv/bin/activate && python3 pair.py"
echo "  3. Start the service: systemctl start the-clicker"
echo "  4. Open dashboard: http://$(hostname -I | awk '{print $1}'):8080"
echo ""
echo "Manage:"
echo "  systemctl status the-clicker"
echo "  systemctl restart the-clicker"
echo "  journalctl -u the-clicker -f"
