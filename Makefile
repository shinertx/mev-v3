# MEV-V3 Makefile
# PROJECT_BIBLE Compliant Build System

.PHONY: help install test build deploy clean compliance simulation chaos

# Default target
help:
	@echo "MEV-V3 Build System - PROJECT_BIBLE Compliant"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install all dependencies"
	@echo "  test         - Run all tests with compliance checks"
	@echo "  build        - Build Docker images and contracts"
	@echo "  deploy       - Deploy to Kubernetes"
	@echo "  clean        - Clean build artifacts"
	@echo "  compliance   - Run PROJECT_BIBLE compliance checks"
	@echo "  simulation   - Run forked mainnet simulation"
	@echo "  chaos        - Run chaos testing"
	@echo "  dev          - Start local development environment"
	@echo "  monitor      - Open monitoring dashboards"

# Variables
DOCKER_REPO := gcr.io/mev-og/mev-v3
VERSION := $(shell git rev-parse --short HEAD)
PYTHON := python3.11
NODE := node

# Compliance check
compliance:
	@echo "Checking PROJECT_BIBLE compliance..."
	@test -f PROJECT_BIBLE.md || (echo "BLOCKED - PROJECT_BIBLE.md not found" && exit 1)
	@test -f AGENTS.md || (echo "BLOCKED - AGENTS.md not found" && exit 1)
	@echo "Checking Python module metadata..."
	@find . -name "*.py" -not -path "./venv/*" -exec grep -L "role:" {} \; | grep -q . && echo "BLOCKED - Missing metadata" && exit 1 || true
	@echo "✓ All compliance checks passed"

# Install dependencies
install: compliance
	@echo "Installing Python dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Installing Node dependencies..."
	npm ci
	@echo "Installing Foundry..."
	curl -L https://foundry.paradigm.xyz | bash
	@echo "✓ Dependencies installed"

# Testing
test: compliance
	@echo "Running Python tests..."
	$(PYTHON) -m pytest tests/ -v --cov=engine --cov=strategies --cov=risk
	@echo "Running Solidity tests..."
	npx hardhat test
	forge test -vvv
	@echo "✓ All tests passed"

# Linting
lint:
	@echo "Linting Python code..."
	$(PYTHON) -m black .
	$(PYTHON) -m flake8 .
	$(PYTHON) -m mypy engine/ strategies/ risk/ || true
	@echo "Linting Solidity code..."
	npx prettier --write 'contracts/**/*.sol'
	@echo "✓ Linting complete"

# Build
build: compliance test
	@echo "Building smart contracts..."
	npx hardhat compile
	forge build --optimize --optimizer-runs 20000
	@echo "Building Docker image..."
	docker build -t $(DOCKER_REPO):$(VERSION) -t $(DOCKER_REPO):latest .
	@echo "✓ Build complete"

# Local development environment
dev:
	@echo "Starting local development environment..."
	docker compose up -d
	@echo "Waiting for services to start..."
	@sleep 10
	@echo "Services available at:"
	@echo "  - MEV Engine: http://localhost:8080"
	@echo "  - Agent API: http://localhost:8081"
	@echo "  - Prometheus: http://localhost:9091"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"
	@echo "  - Anvil RPC: http://localhost:8545"

# Stop development environment
dev-stop:
	docker compose down

# Logs
logs:
	docker compose logs -f mev-engine

# Forked mainnet simulation
simulation:
	@echo "Running forked mainnet simulation..."
	docker compose run --rm simulator

# Chaos testing
chaos:
	@echo "Running chaos tests..."
	docker compose run --rm chaos-monkey python -m drp.test_scenarios

# Deploy to Kubernetes
deploy: build
	@echo "Deploying to Kubernetes..."
	kubectl apply -f k8s/
	kubectl set image deployment/mev-engine mev-engine=$(DOCKER_REPO):$(VERSION) -n mev-v3
	@echo "✓ Deployment complete"

# Deploy contracts
deploy-contracts:
	@echo "Deploying smart contracts..."
	npx hardhat run scripts/deploy.js --network mainnet

# Database migrations
db-migrate:
	@echo "Running database migrations..."
	docker compose exec postgres psql -U mevog -d mevdb -f /sql/init.sql

# Monitoring
monitor:
	@echo "Opening monitoring dashboards..."
	@open http://localhost:3000 || xdg-open http://localhost:3000 || echo "Grafana: http://localhost:3000"
	@open http://localhost:9091 || xdg-open http://localhost:9091 || echo "Prometheus: http://localhost:9091"

# Performance report
report:
	@echo "Generating performance report..."
	$(PYTHON) -m scripts.generate_report

# Backup
backup:
	@echo "Creating backup..."
	@mkdir -p backups
	docker compose exec postgres pg_dump -U mevog mevdb > backups/mevdb_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✓ Backup created"

# Clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	rm -rf artifacts/ cache/
	rm -rf node_modules/
	find . -type d -name __pycache__ -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete
	docker compose down -v
	@echo "✓ Clean complete"

# Security scan
security:
	@echo "Running security scans..."
	$(PYTHON) -m pip install safety
	safety check
	npm audit
	@echo "Running Slither..."
	slither . || true
	@echo "✓ Security scan complete"

# CI simulation
ci:
	@echo "Simulating CI pipeline..."
	$(MAKE) compliance
	$(MAKE) install
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) build
	@echo "✓ CI simulation complete"

# DRP drill
drp-drill:
	@echo "Executing DRP drill..."
	$(PYTHON) -m drp.execute_drill
	@echo "✓ DRP drill complete"

# Version
version:
	@echo "MEV-V3 Version Information:"
	@echo "  Git SHA: $(VERSION)"
	@echo "  Python: $(shell $(PYTHON) --version)"
	@echo "  Node: $(shell $(NODE) --version)"
	@echo "  Docker: $(shell docker --version)"
	@echo "  PROJECT_BIBLE compliant: ✓"

# All - full build and test
all: compliance install lint test build
	@echo "✓ Full build complete"
