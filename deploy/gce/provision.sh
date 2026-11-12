#!/usr/bin/env bash
# Provision the aibom verify REST service on a private GCP VM.
#
# Creates the service account and its roles, the Artifact Registry repository,
# a custom-mode VPC and subnet, a Cloud Router and NAT for egress, firewall
# rules for tcp/8080 from the caller's range and tcp/22 from IAP only, and one
# VM with no external IP that runs deploy/gce/startup.sh on boot.
#
# Re-runnable: every step describes the resource first and creates it only when
# it is missing.
#
# Site-specific values live in deploy/gce/site.env (git-ignored); copy
# deploy/gce/site.env.example and fill it in. Environment variables override it.
#
#   cp deploy/gce/site.env.example deploy/gce/site.env   # once, then edit
#   PUSH_IMAGE=1 ./deploy/gce/provision.sh               # also build and push
#
# Requires: gcloud (authenticated), and docker for PUSH_IMAGE=1.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
if [ -f "$HERE/site.env" ]; then
  # shellcheck source=/dev/null
  . "$HERE/site.env"
fi

PROJECT=${PROJECT:?set PROJECT (or deploy/gce/site.env) to the target GCP project}
REGION=${REGION:-us-east1}
ZONE=${ZONE:-us-east1-b}
NET=${NET:-aibom-vpc}
SUBNET=${SUBNET:-aibom-us-east1}
SUBNET_RANGE=${SUBNET_RANGE:-10.42.0.0/24}
VM=${VM:-aibom-serve}
SA=${SA:-aibom-vm}
VM_TAG=${VM_TAG:-aibom-api}
IMAGE=${IMAGE:-$REGION-docker.pkg.dev/$PROJECT/aibom/aibom-security:0.1.0}
# Source range allowed to reach the API: the calling cluster's node or pod
# egress range. Required, no default.
CLIENT_CIDR=${CLIENT_CIDR:?set CLIENT_CIDR to the calling cluster egress range}
SA_EMAIL=$SA@$PROJECT.iam.gserviceaccount.com

say() { printf '\n== %s\n' "$*"; }

say "project $PROJECT ($REGION/$ZONE)"
gcloud config set project "$PROJECT" >/dev/null

say "enabling APIs"
gcloud services enable compute.googleapis.com artifactregistry.googleapis.com \
  iam.googleapis.com iap.googleapis.com oslogin.googleapis.com --project="$PROJECT"

say "service account $SA"
gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "$SA" --project="$PROJECT" \
       --display-name="aibom verify VM"
for role in roles/artifactregistry.reader roles/logging.logWriter roles/monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:$SA_EMAIL" --role="$role" --condition=None --format='' >/dev/null
done

say "artifact registry"
gcloud artifacts repositories describe aibom --project="$PROJECT" --location="$REGION" >/dev/null 2>&1 \
  || gcloud artifacts repositories create aibom --project="$PROJECT" \
       --repository-format=docker --location="$REGION" \
       --description="aibom-security container images"

if [ "${PUSH_IMAGE:-0}" = "1" ]; then
  say "building and pushing $IMAGE"
  gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
  # Build for the VM's architecture (the build host may be arm64) and without
  # buildx attestation manifests, which COS's dockerd resolves incorrectly.
  docker build --platform=linux/amd64 --provenance=false --sbom=false -t "$IMAGE" .
  docker push "$IMAGE"
fi

say "vpc $NET / subnet $SUBNET"
gcloud compute networks describe "$NET" --project="$PROJECT" >/dev/null 2>&1 \
  || gcloud compute networks create "$NET" --project="$PROJECT" --subnet-mode=custom \
       --bgp-routing-mode=regional
gcloud compute networks subnets describe "$SUBNET" --project="$PROJECT" --region="$REGION" >/dev/null 2>&1 \
  || gcloud compute networks subnets create "$SUBNET" --project="$PROJECT" \
       --network="$NET" --region="$REGION" --range="$SUBNET_RANGE" \
       --enable-private-ip-google-access

say "cloud nat (egress for image pulls and huggingface.co)"
gcloud compute routers describe aibom-nat-router --project="$PROJECT" --region="$REGION" >/dev/null 2>&1 \
  || gcloud compute routers create aibom-nat-router --project="$PROJECT" \
       --network="$NET" --region="$REGION"
gcloud compute routers nats describe aibom-nat --project="$PROJECT" --region="$REGION" \
  --router=aibom-nat-router >/dev/null 2>&1 \
  || gcloud compute routers nats create aibom-nat --project="$PROJECT" \
       --router=aibom-nat-router --region="$REGION" \
       --nat-all-subnet-ip-ranges --auto-allocate-nat-external-ips

say "firewall: API from $CLIENT_CIDR, SSH over IAP only"
gcloud compute firewall-rules describe allow-aibom-api --project="$PROJECT" >/dev/null 2>&1 \
  || gcloud compute firewall-rules create allow-aibom-api --project="$PROJECT" \
       --network="$NET" --direction=INGRESS --action=ALLOW --rules=tcp:8080 \
       --source-ranges="$CLIENT_CIDR" --target-tags="$VM_TAG"
gcloud compute firewall-rules describe allow-iap-ssh --project="$PROJECT" >/dev/null 2>&1 \
  || gcloud compute firewall-rules create allow-iap-ssh --project="$PROJECT" \
       --network="$NET" --direction=INGRESS --action=ALLOW --rules=tcp:22 \
       --source-ranges=35.235.240.0/20 --target-tags="$VM_TAG"

say "vm $VM"
if ! gcloud compute instances describe "$VM" --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
  gcloud compute instances create "$VM" --project="$PROJECT" --zone="$ZONE" \
    --machine-type=e2-standard-2 --no-address \
    --image-family=cos-stable --image-project=cos-cloud \
    --boot-disk-type=pd-balanced --boot-disk-size=50GB \
    --network="$NET" --subnet="$SUBNET" --tags="$VM_TAG" \
    --service-account="$SA_EMAIL" --scopes=cloud-platform \
    --metadata=enable-oslogin=TRUE,aibom-image="$IMAGE" \
    --metadata-from-file=startup-script=deploy/gce/startup.sh \
    --deletion-protection --labels=app=aibom-security,env=poc
else
  gcloud compute instances add-metadata "$VM" --project="$PROJECT" --zone="$ZONE" \
    --metadata=aibom-image="$IMAGE" --format='' >/dev/null
fi

say "state"
gcloud compute instances describe "$VM" --project="$PROJECT" --zone="$ZONE" \
  --format='value(name,status,networkInterfaces[0].networkIP,machineType.basename(),disks[0].diskSizeGb,disks[0].type.basename())'

cat <<EOF

Next:
  1. token  : gcloud compute ssh $VM --zone=$ZONE --tunnel-through-iap \\
                --command 'sudo grep AIBOM_API_TOKEN /mnt/stateful_partition/aibom/serve.env'
  2. check  : gcloud compute ssh $VM --zone=$ZONE --tunnel-through-iap \\
                --command 'curl -sS http://127.0.0.1:8080/healthz'
  3. runs   : /mnt/stateful_partition/aibom/runs on the VM (uid 999)
EOF
