#!/bin/bash
# ═══════════════════════════════════════════════
# ClinicalVR — One-click HTTPS launcher
# Mic works automatically — no permission prompts
# ═══════════════════════════════════════════════

DIR="$(cd "$(dirname "$0")" && pwd)"
CERT="$DIR/localhost.pem"
KEY="$DIR/localhost-key.pem"
PORT=8443

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║       ClinicalVR Launcher            ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# Install mkcert if needed
if ! command -v mkcert &>/dev/null; then
  echo "  Installing mkcert..."
  brew install mkcert && mkcert -install
fi

# Create certificate if needed
if [ ! -f "$CERT" ]; then
  echo "  Creating HTTPS certificate..."
  cd "$DIR" && mkcert localhost
fi

echo "  Starting HTTPS server..."
echo ""
echo "  ┌─────────────────────────────────────────┐"
echo "  │                                         │"
echo "  │  Open Chrome and go to:                 │"
echo "  │                                         │"
echo "  │  https://localhost:8443/ClinicalVR.html  │"
echo "  │                                         │"
echo "  │  Mic works with NO permission prompts   │"
echo "  │                                         │"
echo "  └─────────────────────────────────────────┘"
echo ""
echo "  Press Ctrl+C to stop the server."
echo ""

cd "$DIR"
python3 -c "
import ssl, http.server, sys
cert, key, port = '$CERT', '$KEY', $PORT
class H(http.server.SimpleHTTPRequestHandler):
    def log_message(self, f, *a): pass
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(cert, key)
srv = http.server.HTTPServer(('localhost', port), H)
srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
srv.serve_forever()
"
