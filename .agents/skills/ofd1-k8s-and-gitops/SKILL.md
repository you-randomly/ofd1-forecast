---
name: ofd1-k8s-and-gitops
description: >-
  Guides Docker container builds, GitHub Container Registry (GHCR) CI/CD,
  Kubernetes Kustomize manifests, local 3-node kind cluster testing, and FluxCD
  GitOps deployment for ofd1-forecast. Activate when working on Dockerfile,
  .github/workflows/ghcr-publish.yml, or any manifest in k8s/.
---

# OFD1 Kubernetes, Docker & FluxCD GitOps Skill

This skill covers container packaging, PVC data seeding, and Kubernetes/FluxCD deployment for the `ofd1-forecast` service.

## 1. Container Architecture & PVC Seeding (`Dockerfile`)

- **Base Image:** `ghcr.io/astral-sh/uv:python3.13-bookworm-slim`
- **Runtime User:** Non-root user `caver` (`UID/GID 10001`).
- **Seed Data Pattern (`/app/seed_data` -> `/data`):**
  - The Docker image copies tracked `data/` artifacts (`ofd1_aligned_hourly.parquet`, `ofd1_models_v1.joblib`, `forecast_verification_ledger.parquet`, `phase2_trial_metrics.parquet`, `horizon_breakdown_1h_to_48h.parquet`) into `/app/seed_data` (`OFD1_SEED_DIR`).
  - At pod startup, `OFD1ForecastEngine.initialize()` copies any missing files from `/app/seed_data` into `/data` (`OFD1_DATA_DIR`, mounted from `ofd1-telemetry-pvc`) before starting the 10-minute live sync loop.
  - Because the PVC (`ReadWriteOnce`) is mounted at `/data`, Kubernetes deployments use `strategy.type: Recreate` with `replicas: 1` so volume handoff never deadlocks during rollouts.

## 2. CI/CD Pipeline (`.github/workflows/ghcr-publish.yml`)

- Pushes to `main` or `v*.*.*` tags build `linux/amd64` images via Docker Buildx and publish to:
  `ghcr.io/you-randomly/ofd1-forecast:latest` (plus `sha-<commit>` tags).
- Keep `.dockerignore` updated so large unversioned caches (`data/ofd1_thingspeak_raw.parquet`, `data/historic_ofd_streamway_cleaned.csv`, `data/open_meteo_2015_2023.parquet`, `.venv`, `.agents`, `k8s`, notebooks) are excluded from the container build context.

## 3. Kubernetes & FluxCD Manifests (`k8s/`)

| File | Description |
| :--- | :--- |
| `k8s/kustomization.yaml` | Standard Kustomize entrypoint bundling `namespace.yaml`, `pvc.yaml`, `deployment.yaml`, `service.yaml`, and `httproute.yaml` in namespace `ofd1-forecast`. |
| `k8s/flux-cd/ofd1-forecast.yaml` | All-in-one manifest designed to drop directly into a FluxCD cluster repository (`clusters/my-cluster/apps/ofd1-forecast/ofd1-forecast.yaml`). Includes `Namespace`, Longhorn `PersistentVolumeClaim` (`storageClassName: longhorn-talos`, 2 replicas, `2Gi`), `Deployment`, `Service`, Cilium Gateway `HTTPRoute` (`ofd1.lab.jake-watson.co.uk`), and `DNSEndpoint`. |
| `k8s/flux-cd/ofd1-gitrepo-sync.yaml` | Alternative FluxCD `GitRepository` + `Kustomization` manifest reconciling `./k8s` directly from `https://github.com/you-randomly/ofd1-forecast.git` every 5 minutes. |
| `k8s/kind-3node.yaml` | Local 3-node (`1 control-plane + 2 workers`) `kind` cluster configuration mapping host ports `8080`/`8443`. |

## 4. Validation Commands

1. **Validate Kustomize Build Locally:**
   ```bash
   kubectl kustomize k8s/
   ```
2. **Verify Container Build Locally:**
   ```bash
   docker build -t ghcr.io/you-randomly/ofd1-forecast:local .
   ```
