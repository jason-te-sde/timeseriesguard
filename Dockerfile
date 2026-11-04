# The tsguard scoring service. Build from the repository root: `docker compose up --build`.
#
# The image holds the code and the runtime dependencies only, no model: the bundle is mounted
# read-only at /models (see docker-compose.yml). The `experiment` extra (MLflow, Optuna) and the
# dev group are not installed, so `tsguard serve` must work without them; a unit test keeps
# tsguard.cli and tsguard.serve free of module-level imports of either.
#
# The base images are pinned by tag. Pin each digest the first time the image is pulled
# (`python:3.11-slim@sha256:...` and `ghcr.io/astral-sh/uv:0.12.19@sha256:...`) and record the
# digests in the commit message.

FROM python:3.11-slim AS builder

# The uv version CI uses (setup-uv in .github/workflows/ci.yml).
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv

# Copy, never hard-link: the cache mount below is a different filesystem from the venv.
# Compile the dependencies' bytecode now: the final container's filesystem is read-only, so
# nothing could cache it at run time, and every start would compile torch's modules again.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# The dependencies come first: this layer is rebuilt only when pyproject.toml or uv.lock change.
# On Linux the lockfile resolves torch to the CPU wheel, so no CUDA libraries come with it.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# The project itself. README.md and LICENSE are in pyproject.toml's metadata, so the build backend
# reads them. The project is installed editable, so the venv points at /app/src, where the
# runtime stage puts it again. The files keep the host's permissions, so make them readable for
# the unprivileged user whatever the host's umask was.
COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
    && .venv/bin/python -m compileall -q src \
    && chmod -R a+rX src


FROM python:3.11-slim AS runtime

# An unprivileged user, with a fixed uid so a bind-mounted directory can be made readable for it.
RUN groupadd --gid 10001 tsguard \
    && useradd --uid 10001 --gid tsguard --no-create-home --shell /usr/sbin/nologin tsguard

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

USER tsguard

EXPOSE 8000

# The base image has no curl. This checks port 8000, the one CMD binds; a container started with
# another --port needs its own check. /healthz answers once the bundle is loaded and verified.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

CMD ["tsguard", "serve", "--bundle", "/models/bundle.joblib", "--host", "0.0.0.0", "--port", "8000"]
