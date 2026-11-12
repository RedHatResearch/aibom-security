#!/bin/bash
# Container-Optimized OS startup script for the aibom verify REST service.
#
# Runs on every boot and is idempotent. The image reference comes from the
# `aibom-image` instance metadata key, set at instance creation or updated with
# `gcloud compute instances add-metadata` and a restart.
#
# State on the persistent partition:
#   /mnt/stateful_partition/aibom/runs      -> AIBOM_RUNS_DIR (one dir per run)
#   /mnt/stateful_partition/aibom/cache     -> AIBOM_CACHE_DIR (artifact cache)
#   /mnt/stateful_partition/aibom/serve.env -> bearer token + server env
set -euo pipefail

META=http://metadata.google.internal/computeMetadata/v1/instance/attributes
IMAGE=$(curl -fsS -H 'Metadata-Flavor: Google' "$META/aibom-image")
REGISTRY=${IMAGE%%/*}
STATE=/mnt/stateful_partition/aibom

echo "aibom-startup: image=$IMAGE state=$STATE"

mkdir -p "$STATE/runs" "$STATE/cache" /var/lib/aibom
# The image runs as uid 999 (nonroot), and the server aborts at boot when its
# runs directory is not writable.
chown -R 999:999 "$STATE/runs" "$STATE/cache"

# Bearer token: minted once, kept on the persistent disk, never in git or
# instance metadata. Rotate by deleting the file and rebooting, then updating
# the Secret the Tekton Task reads.
if [ ! -s "$STATE/serve.env" ]; then
  umask 077
  TOKEN=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
  cat >"$STATE/serve.env" <<EOF
AIBOM_API_TOKEN=$TOKEN
AIBOM_RUNS_DIR=/runs
AIBOM_CACHE_DIR=/cache
AIBOM_LOG_LEVEL=INFO
EOF
  echo "aibom-startup: minted AIBOM_API_TOKEN into $STATE/serve.env"
fi
chmod 0600 "$STATE/serve.env"

# Health gate run as ExecStartPost; the unit fails when the API does not answer.
cat >/var/lib/aibom/healthz <<'EOF'
#!/bin/bash
for _ in $(seq 1 30); do
  if /usr/bin/docker exec aibom-serve python -c \
    'import urllib.request as u; u.urlopen("http://127.0.0.1:8080/healthz", timeout=3)' \
    2>/dev/null; then
    exit 0
  fi
  sleep 2
done
echo "aibom-healthz: no healthy response after 60s" >&2
exit 1
EOF
chmod 0755 /var/lib/aibom/healthz

cat >/etc/systemd/system/aibom-serve.service <<UNIT
[Unit]
Description=aibom verify REST service
Wants=network-online.target
After=network-online.target docker.service
Requires=docker.service

[Service]
Environment=HOME=/var/lib/aibom
ExecStartPre=/usr/bin/docker-credential-gcr configure-docker --registries $REGISTRY
ExecStartPre=-/usr/bin/docker rm -f aibom-serve
ExecStartPre=/usr/bin/docker pull $IMAGE
ExecStart=/usr/bin/docker run --rm --name=aibom-serve \\
  --env-file $STATE/serve.env \\
  -p 8080:8080 \\
  -v $STATE/runs:/runs -v $STATE/cache:/cache \\
  --log-opt max-size=10m --log-opt max-file=3 \\
  --entrypoint aibom-serve $IMAGE \\
  --host 0.0.0.0 --port 8080 --concurrency 2 --queue-size 500
ExecStartPost=/bin/bash /var/lib/aibom/healthz
ExecStop=/usr/bin/docker stop aibom-serve
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable aibom-serve
systemctl restart aibom-serve
echo "aibom-startup: done"
