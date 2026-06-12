.PHONY: dev check test test-integration bench-concurrency up down

UV ?= uv
RUN ?= $(UV) run --extra dev --extra test
PYTHON ?= $(RUN) python
RUFF ?= $(RUN) ruff
MYPY ?= $(RUN) mypy
PYTEST ?= $(RUN) pytest

dev:
	@set -e; \
	TIDEGATE_ADMIN_TOKEN=dev-admin MOCK_A_KEY=mock-key \
	$(PYTHON) -m mock_provider --port 9001 & mock_pid=$$!; \
	TIDEGATE_ADMIN_TOKEN=dev-admin MOCK_A_KEY=mock-key \
	$(PYTHON) -m tidegate --config config/gateway.yaml & gate_pid=$$!; \
	trap 'kill $$mock_pid $$gate_pid 2>/dev/null || true' INT TERM EXIT; \
	wait

check:
	$(RUFF) format --check .
	$(RUFF) check .
	$(MYPY) .

test:
	$(PYTEST) tests/unit

test-integration:
	$(PYTEST) -m integration tests/integration

bench-concurrency:
	@set -e; \
	mkdir -p out; \
	echo "ulimit -n before: $$(ulimit -n)"; \
	ulimit -n 16384 || true; \
	echo "ulimit -n after: $$(ulimit -n)"; \
	pid_arg=""; \
	if [ -n "$${TIDEGATE_GATEWAY_PID:-}" ]; then pid_arg="--gateway-pid $${TIDEGATE_GATEWAY_PID}"; fi; \
	$(PYTHON) scripts/loadgen.py \
		--scenario concurrency \
		--rps $${TIDEGATE_BENCH_RPS:-20} \
		--duration $${TIDEGATE_BENCH_DURATION:-180} \
		--stream-ratio 1.0 \
		--mock-tpot-ms $${TIDEGATE_BENCH_TPOT_MS:-300} \
		--mock-output-tokens $${TIDEGATE_BENCH_OUTPUT_TOKENS:-100} \
		$$pid_arg \
		--output out/concurrency.json

up:
	docker compose -f deploy/docker-compose.yml up -d

down:
	docker compose -f deploy/docker-compose.yml down
