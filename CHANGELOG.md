# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/); before 1.0, the bundle format and the `metrics.json`
schema may change between minor versions.

## Unreleased

### Changed

- Runtime dependencies: newer patch and minor releases of pandas, fastapi and pydantic. Dependabot
  moves them weekly; the versions a release ships are the ones in its `uv.lock`.
- Base image: a newer `python:3.11-slim` build, pinned by digest in the `Dockerfile`.

## 0.1.0 — 2026-11-09

### Added

- SMD access pinned to one OmniAnomaly commit: a checksum manifest for every file, a verified fetch
  that never leaves a partial file, and a six-rule data contract (`tsguard data fetch`,
  `tsguard data verify`).
- Test labels that can be read only in the final-evaluation phase, with a log of every read, and
  thresholds that can be frozen only before it.
- Walk-forward folds with a 64-row purge, and checks that re-derive the fold and split invariants
  from their numbers.
- 63 named feature specs (59 per channel, 4 calendar; 2,246 columns), each a pure function of a
  trailing window of at most 64 rows, and a feature pipeline that standardizes with statistics
  fitted once, floors every scale at 1e-3, clips to ±50, and refuses to evaluate its own fit rows.
- The future-perturbation tests for transforms (`assert_causal`) and for fitting
  (`assert_fit_ignores_future`).
- Four detectors behind one protocol: a robust z-score, Isolation Forest, and 1D-CNN and LSTM
  autoencoders that train deterministically on CPU.
- Four evaluation protocols (point-wise, point adjust, PA%K, event-level with `L = 60`, `C = 60`)
  and a test that shows random scores beat point adjust.
- Seeded synthetic anomaly injection, and label-free selection: thresholds, hyper-parameters
  (Optuna, recorded in MLflow) and the headline model are chosen on injected anomalies.
- The FFT ablation over five seeds.
- `tsguard reproduce`, which runs the whole experiment in a fixed phase order and writes
  `metrics.json`, with a checkpoint before the ablation; `tsguard claims`, which derives the README
  numbers from it and checks them.
- A model bundle with a sha256 sidecar and a feature-code fingerprint, a FastAPI service that scores
  one row within a relative 1e-6 of the batch score (`tsguard bundle`, `tsguard serve`), a
  container, and a latency measurement script against a p95 budget of 50 ms.
- Ten injected leakage defects, each asserted to be caught by its named guard, and a determinism
  test that runs the experiment twice.
- CI workflows for lint, tests on Python 3.11 and 3.12, leakage, protocol, determinism, smoke,
  container, claims, a clean-clone build and commit hygiene, and a manual full-run workflow. All
  but the full-run workflow have passed on GitHub's hosted runners.
- `reports/`: the full 28-machine run (2 h 56 min on an Apple M2), its 144 tracked runs, the p95
  latency of the served headline model, and the README block derived from them.
- Documentation: architecture, evaluation protocol, testing, operations, four decision records.
