.PHONY: dev check test test-integration up down

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
	$(PYTEST) -m integration tests/integration/test_m0_e2e.py

up:
	docker compose -f deploy/docker-compose.yml up -d

down:
	docker compose -f deploy/docker-compose.yml down
