# The full run on a hosted Linux runner

`metrics.json` and `mlflow_runs.csv` here are the artifacts of
[run 44461761747](https://github.com/jason-te-sde/timeseriesguard/actions/runs/44461761747) of
`.github/workflows/full-run.yml`, started by hand on 2026-11-19 at commit 1e91a3f. The runner was
GitHub's hosted `ubuntu-latest` for a public repository (x86_64, 4 vCPUs, 16 GB); `environment` in
`metrics.json` records the platform and the library versions. The reproduce step took 2 h 16 min.

The committed run in `reports/` was made on an Apple M2. Neither file here feeds the README's
results block or `tsguard claims`: they are evidence that the run can be repeated elsewhere, and
how closely.

To compare the two:

```bash
uv run python - <<'PY'
import json
a = json.load(open("reports/linux-run/metrics.json"))
b = json.load(open("reports/metrics.json"))
print(a["headline"]["model"], a["headline"]["auc_pr"], b["headline"]["model"], b["headline"]["auc_pr"])
worst = max(
    abs(a["models"][m]["macro"][p]["auc_pr"] - b["models"][m]["macro"][p]["auc_pr"])
    for m in a["models"]
    for p in a["models"][m]["macro"]
)
print("largest macro AUC-PR difference:", round(worst, 5))
PY
```
