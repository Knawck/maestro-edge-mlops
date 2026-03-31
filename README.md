# MAESTRO

### Multi-path Adaptive Edge System for Temporal Real-time Orchestration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![MEXT Research 2026](https://img.shields.io/badge/MEXT-Research%202026-red.svg)]()
[![All Tools Free](https://img.shields.io/badge/tools-100%25%20free%20%2F%20open--source-green.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

> **Self-Healing Edge MLOps for Real-Time Generative Audio Systems**
>
> A portfolio of 7 Proof-of-Concept modules demonstrating every core idea
> in my MEXT 2026 research proposal — built on a Windows laptop, with no
> cloud account, no paid software, and no prior Japan access required.

---

## The Problem

Real-time musical collaboration across cities requires:

- Round-trip latency **< 20ms**
- Jitter **< 2ms**
- Recovery from failure **< 500ms**

No existing platform achieves this reliably. The reason is architectural:
they treat the network as a static constraint rather than a dynamic resource
to be managed by intelligent, self-healing infrastructure.

MAESTRO is the architecture that fixes this.

---

## Who built this

**Anirban Biswas**
Cloud & DevOps Engineer (TCS · Infosys) · Music Producer & Composer
Kolkata, India · [myself.anirban11@gmail.com](mailto:myself.anirban11@gmail.com)
[linkedin.com/in/anirban-biswas11](https://linkedin.com/in/anirban-biswas11)

MEXT Scholarship Application 2026 · Target: Tokyo Institute of Technology / UTokyo
Research domain: Edge MLOps · Real-Time Audio AI

---

## Architecture

MAESTRO is a six-layer architecture. Each PoC below maps directly to one layer
and proves one concrete research claim.

| PoC | MAESTRO Layer | Research Claim Demonstrated | Status |
|-----|--------------|----------------------------|--------|
| [PoC-01](./poc-01-audio-latency-lab) | Layer 1: Edge Audio Ingestion | Jitter > 8ms destroys musical timing coherence | ✅ Complete |
| [PoC-02](./poc-02-lstm-quality-predictor) | Layer 2: ML Quality Management | LSTM predicts degradation 200ms ahead, < 5ms inference | 🔄 Building |
| [PoC-03](./poc-03-musical-sync-protocol) | Layer 3: Musical Sync Protocol | Beat-grid UDP sync survives jitter injection | 🔄 Building |
| [PoC-04](./poc-04-multipath-fec-transport) | Layer 4: Multi-Path FEC | Reed-Solomon maintains continuity at 20% packet loss | 🔄 Building |
| [PoC-05](./poc-05-recovery-controller) | Layer 5: Recovery Controller | Joint FSM recovers in < 500ms across both layers | 🔄 Building |
| [PoC-06](./poc-06-edge-mlops-lifecycle) | Layer 6: Edge MLOps | Zero-downtime model update on Kubernetes | 🔄 Building |
| [PoC-07](./poc-07-observability-plane) | Cross-cutting: Observability | Unified Prometheus + Grafana across all layers | 🔄 Building |

---

## Quick Start (Windows)
```powershell
git clone https://github.com/anirban-biswas/maestro-edge-mlops
cd maestro-edge-mlops\poc-01-audio-latency-lab
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python src/latency_measure.py --duration 30 --out results/clean.csv
python src/visualise_results.py
```

---

## Tools used — all free and open-source

| Tool | Purpose | License |
|------|---------|---------|
| Python 3.11 | All PoC code | PSF |
| PyTorch (CPU) | LSTM model | BSD-3 |
| FastAPI + uvicorn | Inference API | MIT |
| MLflow | Experiment tracking + model registry | Apache 2.0 |
| DVC | Training data versioning | Apache 2.0 |
| Docker Desktop | Containerisation + local Kubernetes | Apache 2.0 engine |
| Prometheus | Metrics collection | Apache 2.0 |
| Grafana OSS | Dashboards | AGPL v3 |
| ArgoCD | GitOps continuous deployment | Apache 2.0 |
| Terraform | Infrastructure as Code | BSL |
| Oracle Cloud | Cloud VM (always-free tier) | — |
| Clumsy 0.3 | Windows network degradation | GPL v3 |

---

## Repository structure
```
maestro-edge-mlops/
├── README.md                          ← you are here
├── ARCHITECTURE.md                    ← maps each PoC to its research claim
├── docs/
│   └── maestro-research-plan.pdf      ← full MEXT research proposal
│
├── poc-01-audio-latency-lab/          ← ✅ complete
├── poc-02-lstm-quality-predictor/     ← 🔄 building
├── poc-03-musical-sync-protocol/      ← 🔄 building
├── poc-04-multipath-fec-transport/    ← 🔄 building
├── poc-05-recovery-controller/        ← 🔄 building
├── poc-06-edge-mlops-lifecycle/       ← 🔄 building
├── poc-07-observability-plane/        ← 🔄 building
│
├── .github/
│   └── workflows/                     ← GitHub Actions CI (Phase 2)
│
└── infra/
    └── terraform/                     ← Oracle Cloud IaC (Phase 2)
```

---

## Build phases

**Phase 1 — Research PoCs (now)**
Get all 7 PoCs running locally. Each proves one claim from the research plan.

**Phase 2 — Engineering layer (after Phase 1)**

| Phase | What | Tools | Time |
|-------|------|-------|------|
| 2a | GitHub Actions CI on all 7 PoCs | GitHub Actions | 1 afternoon/PoC |
| 2b | DVC data versioning + MLflow staging | DVC, MLflow | 1 weekend |
| 2c | Deploy PoC-06 + PoC-07 to Oracle Cloud | Oracle Always Free | 1 weekend |
| 2d | ArgoCD GitOps auto-deployment | ArgoCD | 1 weekend |
| 2e | Terraform — provision Oracle infra as code | Terraform | 1 afternoon |

---

## Research context

The MAESTRO framework proposes a unified self-healing Edge MLOps architecture
in which AI models manage the reliability of real-time audio streams, and the
MLOps infrastructure manages the reliability of those AI models.

This portfolio is the home-scale proof that the architecture is implementable.
The Japan inter-city deployment across Tokyo–Osaka–Fukuoka is the research.
This is the foundation it stands on.

---

*MEXT Scholarship Application 2026 · Anirban Biswas ·
```

---

## Final folder structure check

After all the above, your folder should look exactly like this:
```
D:\maestro-edge-mlops\
├── README.md
├── docs\
│
└── poc-01-audio-latency-lab\
    ├── README.md
    ├── requirements.txt
    ├── src\
    │   ├── latency_measure.py
    │   ├── visualise_results.py
    │   └── clumsy_guide.md
    └── results\
        ├── clean.csv          ← created when you run the script
        ├── degraded.csv       ← created when you run with Clumsy
        └── comparison.png     ← created by visualise_results.py