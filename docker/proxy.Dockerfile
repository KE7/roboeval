# docker/proxy.Dockerfile
# ─────────────────────────────────────────────────────────────────────
# VLA Round-Robin Proxy — lightweight HTTP reverse proxy for VLA backends.
#
# Distributes /predict requests round-robin across N identical VLA backend
# servers with health checking and automatic failover. No GPU needed.
#
# Build (from project root):
#   docker build -f docker/proxy.Dockerfile -t robo-eval/proxy:latest .
#
# Run:
#   docker run --rm -p 5200:5200 robo-eval/proxy:latest \
#       --backends http://host.docker.internal:5100 http://host.docker.internal:5101
#
# NOTE: Use host.docker.internal (Docker Desktop) or --network host (Linux)
# to reach VLA backends running on the host or in sibling containers.
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="robo-eval" \
      description="VLA round-robin proxy (lightweight, no GPU)"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ── Python deps (minimal — no GPU, no rendering) ─────────────────────
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        httpx

# ── Copy proxy code ─────────────────────────────────────────────────
COPY robo_eval/__init__.py /app/robo_eval/
COPY robo_eval/proxy.py /app/robo_eval/

EXPOSE 5200

ENTRYPOINT ["python", "-m", "robo_eval.proxy"]
CMD ["--port", "5200", "--backends", "http://host.docker.internal:5100"]
