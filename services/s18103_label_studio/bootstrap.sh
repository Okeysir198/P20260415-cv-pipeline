#!/usr/bin/env bash
# bootstrap.sh — finish Label Studio setup after first `docker compose up -d`.
#
# Handles two LS 1.23+ quirks that env vars alone can't fix:
#   1. legacy_api_tokens_enabled is a per-organization DB flag that defaults
#      to False even when LABEL_STUDIO_LEGACY_API_TOKEN_AUTH_ENABLED=true.
#   2. The bootstrap admin (LABEL_STUDIO_USERNAME/PASSWORD) is only created
#      on a fresh volume — after `docker compose down -v`.
#
# Outputs the admin API token on success. Safe to re-run.
#
# Usage:
#   cd services/s18103_label_studio
#   docker compose up -d
#   ./bootstrap.sh

set -euo pipefail

CONTAINER="s18103_label_studio-label-studio-1"
HEALTH_URL="http://localhost:18103/health"

# Wait for the service to be healthy (max 90s).
for _ in $(seq 1 30); do
  if curl -sf --max-time 2 "$HEALTH_URL" 2>/dev/null | grep -q UP; then
    break
  fi
  sleep 3
done
curl -sf --max-time 2 "$HEALTH_URL" | grep -q UP || {
  echo "Label Studio not healthy at $HEALTH_URL" >&2
  exit 1
}

# Enable legacy API tokens on every organization, print admin token.
docker exec "$CONTAINER" bash -c 'cd /label-studio/label_studio && \
  DJANGO_SETTINGS_MODULE=core.settings.label_studio python - << "PY" 2>/dev/null
import django; django.setup()
from organizations.models import Organization
from rest_framework.authtoken.models import Token
from users.models import User

for o in Organization.objects.all():
    if hasattr(o, "jwt"):
        o.jwt.legacy_api_tokens_enabled = True
        o.jwt.api_tokens_enabled = True
        o.jwt.save()

for u in User.objects.filter(is_superuser=True):
    tok = Token.objects.filter(user=u).first()
    if tok:
        print(f"ADMIN_EMAIL={u.email}")
        print(f"ADMIN_TOKEN={tok.key}")
PY
'
