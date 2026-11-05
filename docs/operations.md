# Operations

For whoever is woken up by this service. Each section starts with the command and ends with what to
do when it does not work.

## Run the service

The service serves one **bundle**: a fitted feature pipeline, a fitted detector and the threshold
frozen for them, for one machine, in one file with a sha256 sidecar next to it. Build one, then
serve it, in a container or directly:

```sh
uv run tsguard data fetch --entities machine-1-1
make bundle                    # artifacts/bundle/bundle.joblib and bundle.joblib.sha256
make container-up              # docker compose up --build --wait; stop with docker compose down
# or, without Docker:
uv run tsguard serve --bundle artifacts/bundle/bundle.joblib
```

`make bundle` runs `tsguard bundle --entity machine-1-1 --metrics reports/metrics.json`: the
headline model of the committed full run, with its tuned parameters.
`make bundle BUNDLE_ARGS="--model iforest --default-params"` builds a bundle without a metrics file.
Building reads no label: the bundle is the experiment's own selection, run once.

`tsguard bundle` selects with `RunConfig()`, the default settings, and that is exactly the run
configuration of the full profile (seed, three folds, the causality checks on, eight epochs for the
autoencoders; the test `test_the_bundle_is_selected_with_the_config_of_the_full_profile` pins
`FULL.run == RunConfig()`). A bundle built from `reports/metrics.json` therefore repeats, for one
machine, the selection the full run made with the same tuned parameters.

The container runs `tsguard serve --bundle /models/bundle.joblib --host 0.0.0.0 --port 8000` as an
unprivileged user (uid 10001) on a read-only root filesystem. `docker-compose.yml` mounts
`artifacts/bundle/` read-only at `/models` and publishes the port on `127.0.0.1` only, because the
service has no authentication; both bundle files must be readable by uid 10001. Run directly,
`tsguard serve` binds `127.0.0.1:8000` unless `--host` and `--port` say otherwise. The image has been
built and run on an Apple M2 (Docker Desktop, linux/arm64): it starts healthy in under a minute, runs
as uid 10001, cannot write its root filesystem, and has neither MLflow nor Optuna installed. The
`container` CI job builds it on linux/amd64 and serves it on a hosted runner; on the first pull
request it came up healthy and measured a p95 of 14.7 ms against the 50 ms budget.

The bundle is loaded and verified before the server listens, so a missing, damaged or stale bundle
stops the process at start instead of serving errors. The process runs torch on one thread and the
Isolation Forest with one job, so one request never fans out across the machine.

Check it:

```sh
curl -s http://127.0.0.1:8000/healthz       # {"status":"ok"}
curl -s http://127.0.0.1:8000/v1/model      # entity, model, threshold, required_rows, fit_end, bundle_sha256, ...
```

### What a score request looks like

`POST /v1/score` scores the **last** row of `history`. To score row `r` of a machine's series:

```text
{"entity": "machine-1-1", "first_index": r - required_rows + 1, "history": [required_rows rows of 38 numbers]}
```

- `history` is the raw channel values, oldest row first, 38 per row. The service computes the
  features itself.
- `first_index` is the absolute row index of `history[0]` in the machine's full series, counting from
  the first train row. The calendar features depend on it, so it must be right: the scored row is
  `first_index + len(history) - 1`, and test row `i` of a machine is row `train_end + i`.
- Send exactly `required_rows` rows (from `GET /v1/model`; 64 for the feature models, the window
  length for an autoencoder). More rows are accepted, up to 10,000, but only the last
  `required_rows` are read; the rest only costs parsing time.

The response carries `score`, `threshold`, `is_anomaly` (`score >= threshold`), `row_index` and the
`bundle_sha256` of the file that produced it. The score matches the score the batch evaluation gave
the same row within a relative 1e-6 (tested for a feature model and an autoencoder); a reloaded
bundle's batch scores over the test rows are bit-identical to the experiment's.

## What each error means and what to send instead

Every refusal says what to send instead. None of them echoes the request back.

| Status | `detail` | Meaning | Send instead |
| --- | --- | --- | --- |
| 404 | `{"message", "served_entity"}` | The request names a machine this process does not serve. | Send it to the service of that machine. If the rows really are from `served_entity`, fix the `entity` field. |
| 422 | `{"message": "needs N rows of history, got M", "required_rows", "received_rows"}` | Too few rows to compute the features and the detector window of the last row. | At least `required_rows` rows ending at the row you want scored. |
| 422 | `{"message", "fit_end"}` | The scored row lies inside the rows the feature scaling was fitted on. Scoring it would grade the model on its own fitting data (guard `fit_end`). | Score rows from `fit_end` on. Earlier rows may still appear in `history` as context. |
| 422 | `[{"type", "loc", "msg"}, ...]` | Not a valid request. `loc` points at the field. | Fix that field: every row has 38 numbers; values are finite with magnitude at most `1e6` (the data itself is in [0, 1]); `history` has 1 to 10,000 rows; `first_index` is an integer from 0 to `2**53 - 10000`; numbers are JSON numbers, not strings or booleans; no field beyond `entity`, `first_index` and `history`. |

A 404 whose `detail` is the string `"Not Found"` is a wrong path, not a wrong machine, and a 405
(`"Method Not Allowed"`) is the right path with the wrong method, such as a `GET /v1/score`. The
routes are `GET /healthz`, `GET /v1/model` and `POST /v1/score`.

**Any 500 is a bug.** Every input found so far to reach a 500 (huge magnitudes, a huge `first_index`,
windows of tiny magnitudes) is now refused by the schema or handled by the features, so a 500 means a
new case. Keep the request body and the server log, and open an issue.

## The process will not start

`tsguard serve` prints one line, `tsguard: error: <message>`, and exits with code 1. In the container
that line is in `docker compose logs tsguard`, and `docker compose up --wait` reports the service as
not healthy.

| Message contains | Cause | Fix |
| --- | --- | --- |
| `No such file or directory` | No file at `--bundle` (in the container: nothing at `artifacts/bundle/bundle.joblib`). | Run `make bundle`, or fix the path. |
| `no checksum file` | The `.sha256` sidecar is not next to the bundle. | Copy both files together; never write a sidecar by hand. |
| `sha256 is <x> but bundle.joblib.sha256 says <y>` | The file changed after it was written: damaged, truncated or replaced. | Rebuild the bundle. Do not edit the sidecar to match. |
| `the feature code computes different values than when this bundle was built` | The code that computes features changed since the bundle was built, so the scaling stored in it belongs to the old features. | Rebuild the bundle with the code you are running. |
| `not a tsguard bundle of format 2` | A bundle from another version of tsguard, or not a bundle at all. | Rebuild it. |
| `does not reproduce the state recorded`, or a pipeline-state error ending in `rebuild the bundle` | The pipeline or the detector read from the file does not match the digest recorded when it was saved. | Rebuild it. |
| `Permission denied` (container) | The bundle files are not readable by uid 10001. | Make them world-readable, or owned by that uid. |

## Verify a bundle

The sidecar holds the sha256 of the bundle file as 64 lowercase hex characters and a newline. It is
not in `sha256sum`'s `digest  filename` format, so compare the digests directly:

```sh
expected="$(cat artifacts/bundle/bundle.joblib.sha256)"
actual="$(shasum -a 256 artifacts/bundle/bundle.joblib | cut -d ' ' -f 1)"   # sha256sum on Linux
[ "$expected" = "$actual" ] && echo match || echo MISMATCH
curl -s http://127.0.0.1:8000/v1/model    # bundle_sha256: the file the running process loaded
```

The service checks this itself before it unpickles anything, then checks three more things: the
pipeline's state digest, the detector's state digest, and a fingerprint of the feature code (the
features of a fixed synthetic window, recomputed at load and compared with a relative tolerance of
1e-6, so a bundle built on macOS loads on Linux where the last bit of a libm result may differ).

The sidecar proves integrity, not origin: whoever can write the bundle can write its sidecar, and
unpickling a bundle runs code. Load only bundles you built.

## Regenerate a bundle

Rebuild when the service refuses the bundle, after a new `reports/metrics.json`, or whenever you are
not sure where a bundle came from:

```sh
rm -f artifacts/bundle/bundle.joblib artifacts/bundle/bundle.joblib.sha256
make bundle   # the headline model of reports/metrics.json
```

It prints the file, the machine, the model and the new digest. Building needs the machine's data in
the cache and runs the same selection the experiment ran (three folds, injected anomalies, the
causality checks); it reads no label. Restart the service afterwards (`docker compose restart`, or
restart `tsguard serve`): a process loads its bundle once, at start.

## Latency budget

The budget is a p95 below **50 ms** per score request (`P95_BUDGET_MS` in `tsguard.repro.claims`).
What is measured is the client-side HTTP round trip: the time from a ready request body to the
fully read response, so it includes the service's JSON parsing, scoring and response and the hop to
it, not the scoring time alone. Requests are sequential over one keep-alive connection, after a
warm-up. `scripts/latency.py` measures it (`make latency` writes `artifacts/latency.json`):

```sh
uv run python scripts/latency.py --n 2000 --warmup 100 --budget-ms 50 --out reports/latency.json
```

It asks `/v1/model` for the machine and the rows a request needs, loads that machine's whole series
(train and test rows, no labels), and sends request `i` for the `required_rows` rows ending at test
row `i`, wrapping around the test range, so the first requests reach back into the train rows. It
reports p50, p95, p99 and the maximum, and records the `model` and `bundle_sha256` the service
reported, so the file says which service it measured. It exits 1 if the p95 is not below the
budget, if any response is not a 200, or if the service cannot be reached. CI's `container` job runs it with the
same arguments against the container and writes `artifacts/latency-ci.json`, which it uploads.
`tsguard claims --write` puts the `p95_ms` of `reports/latency.json` next to the budget in
`claims.json` and the README row "p95 HTTP round trip, client-side (ms)". It refuses a
`latency.json` whose `model` is not the headline model, or that names none: serve the headline
model (`make bundle`) and measure again. The committed `reports/latency.json` was measured on the
M2, client and container on the same machine, against the headline bundle for machine-1-1.

If the p95 is over budget:

- Check that clients send `required_rows` rows and not thousands: parsing scales with the rows sent.
- A feature model computes 2,246 feature columns over 64 rows per request; an autoencoder scores one
  window. Compare with the model in `/v1/model` before blaming the network.
- The process uses one thread on purpose. Do not raise it to buy latency: scores are only
  reproducible at a fixed thread count.

## `tsguard data fetch` reports a checksum mismatch

The dataset comes from one fixed commit of the OmniAnomaly repository
(`7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4`), and `src/tsguard/data/smd_manifest.json` records the
size and sha256 of every file at that commit. Content at a commit cannot change. A mismatch therefore
means the bytes were corrupted (on disk or in transit) or tampered with on the way, for example by a
proxy or a captive portal that answered with an HTML page.

1. **Never update the manifest to match.** It is the record of what every reported number was
   computed on. A manifest edited to fit a download makes every number unverifiable.
2. **Delete the bad file and fetch again.** The cache is `$TSGUARD_DATA_DIR`, by default
   `~/.cache/tsguard/smd/7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4`. Which file is bad depends on who
   reported it:
   - A mismatch from `tsguard data verify`, from `load_entity` or from a `data` test is about the
     cache. The file named in the message (`<path> in <dir>: expected ..., got ...; deleted`) has
     already been deleted, and `verify` stops at the first one it finds.
   - A mismatch from `tsguard data fetch` is about a fresh download. `fetch` re-downloads any cached
     file that does not match, writes each download to `<file>.part`, verifies it and only then moves
     it into place; a download that does not match is deleted and reported as
     `<path>: expected <n> bytes with sha256 <x>, got <m> bytes with sha256 <y>`. Nothing bad enters
     the cache.

   Either way, the next step is the same:

   ```sh
   uv run tsguard data fetch      # downloads what is missing or wrong, verifies, moves into place
   uv run tsguard data verify     # all files of all 28 machines against the manifest
   ```

3. If a fresh download still does not match, look at what the network returns for the `<path>` in
   the message:

   ```sh
   curl -sS "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4/ServerMachineDataset/<path>" | head -c 300
   ```

   HTML instead of comma-separated numbers is a proxy or a portal; retry from another network.
4. If GitHub itself serves bytes that do not match the pinned commit, stop and report it. Do not run
   experiments on the data until it is explained.

A `FetchError` is different: the download did not complete at all (DNS, timeout, HTTP error). The
message names the file and the URL. CI retries a fetch three times before failing.

## The long experiment

`make reproduce` runs the full profile into `reports/`; it takes hours. Progress goes to stderr, one
line per phase, entity and model, each with a UTC time and the seconds since start, so the run can be
followed from a log file:

```sh
uv run tsguard reproduce --profile full --out reports 2> reproduce.log
```

- It refuses to start while `reports/mlruns.db` exists, because the export reads the whole MLflow
  store and a second run would duplicate rows. Delete the store or pass a fresh `--tracking-uri`.
- It writes `reports/metrics.partial.json` before the ablation, the last and longest phase. If the run
  fails in the ablation, that file holds everything before it (profile `full-partial`).
  `metrics.json` replaces it on success.
- One machine's runner holds 2 to 3 GB of cached features and is released before the next machine is
  loaded. On a 16 GB machine, close other memory-heavy processes first.
- When it ends, stop every process you started for it.
