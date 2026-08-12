#!/bin/bash
# Auto-download & connect to a free VPN via OpenVPN
# OpenVPN da duoc cai san. Script nay se download config tu vpngate.

set -e

echo "=== Downloading VPN config from vpngate.net ==="
CONFIG_DATA=$(curl -s "https://www.vpngate.net/api/iphone/" 2>/dev/null | head -3 | tail -1)
VPN_CONFIG=$(echo "$CONFIG_DATA" | base64 -d 2>/dev/null)

if [ -z "$VPN_CONFIG" ]; then
    echo "ERROR: Cannot download VPN config"
    exit 1
fi

CONFIG_FILE="/tmp/vpngate_vpn.ovpn"
echo "$VPN_CONFIG" > "$CONFIG_FILE"
echo "OK - Config saved to $CONFIG_FILE"

echo ""
echo "=== Connecting (need sudo) ==="
echo "Run: sudo openvpn --config $CONFIG_FILE"
echo ""
echo "After connected, open a new terminal and run:"
echo "  cd $(pwd)"
echo "  python scripts/extract_transcripts_from_media_info.py"
echo ""
echo "When done, Ctrl+C to disconnect VPN."
