#!/usr/bin/env bash
# Verify the 4 services this skill requires are up.
# Exits non-zero on the first DOWN service so callers can stop the pipeline.

set -u

declare -a SERVICES=(
  "SAM3:18100:/health"
  "LabelStudio:18103:/health"
  "AutoLabel:18104:/health"
  "AnnotationQA:18105:/health"
)

fail=0
for s in "${SERVICES[@]}"; do
  IFS=':' read -r name port path <<<"$s"
  if curl -sf --max-time 3 "http://localhost:${port}${path}" >/dev/null; then
    printf "  %-14s :%s  UP\n" "$name" "$port"
  else
    printf "  %-14s :%s  DOWN\n" "$name" "$port"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo ""
  echo "One or more required services are DOWN. Start them with:"
  echo "  cd services/s18100_sam3_service          && docker compose up -d && cd ../.."
  echo "  cd services/s18103_label_studio          && docker compose up -d && ./bootstrap.sh && cd ../.."
  echo "  cd services/s18104_auto_label            && docker compose up -d && cd ../.."
  echo "  cd services/s18105_annotation_quality_assessment && docker compose up -d && cd ../.."
  exit 1
fi
