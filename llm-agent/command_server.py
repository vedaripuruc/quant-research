#!/usr/bin/env python3
"""
Curupira Command Center — HTTP server
Serves data/ directory on port 8043 with CORS and cache-busting headers.
Usage: python command_server.py [--port 8043]
"""

import http.server
import os
import sys
import argparse
from functools import partial

PORT = 8043
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class CommandCenterHandler(http.server.SimpleHTTPRequestHandler):
    """Serves files from data/ with CORS and no-cache headers."""

    def end_headers(self):
        # CORS for local dev
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        # No cache — data refreshes every 10s
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # Quieter logging — only errors and non-200s
        if len(args) >= 2 and "200" in str(args[1]):
            return
        super().log_message(format, *args)

    def guess_type(self, path):
        """Add JSONL mime type."""
        if path.endswith(".jsonl"):
            return "application/x-jsonlines"
        return super().guess_type(path)


def main():
    parser = argparse.ArgumentParser(description="Curupira Command Center server")
    parser.add_argument("--port", "-p", type=int, default=PORT, help=f"Port (default: {PORT})")
    parser.add_argument("--bind", "-b", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    handler = partial(CommandCenterHandler, directory=DATA_DIR)
    server = http.server.HTTPServer((args.bind, args.port), handler)

    print(f"🌿 Curupira Command Center")
    print(f"   Serving: {DATA_DIR}")
    print(f"   URL:     http://localhost:{args.port}/command_center.html")
    print(f"   Port:    {args.port}")
    print(f"   Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
