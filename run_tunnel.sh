#!/bin/bash
set -e

cd "$(dirname "$0")"
exec cloudflared tunnel --protocol http2 --url http://127.0.0.1:8000
