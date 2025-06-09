# MEV-V3 Trading Engine

**PROJECT_BIBLE Compliant** | **Mutation-Ready** | **Zero Human Ops**

A sovereign, adversarial-grade, AI-native MEV trading engine designed to compound $5K→$10M+ through automated arbitrage, flash loans, and liquidations on Ethereum.

## 🚨 Critical Requirements

This system operates under strict PROJECT_BIBLE.md governance. All code, operations, and modifications must comply with the canonical rules defined therein. Non-compliance will result in automatic blocking.

**Key Thresholds:**
- Sharpe Ratio: ≥ 2.5
- Max Drawdown: ≤ 7%
- Uptime: ≥ 95%
- Median PnL: ≥ Gas × 1.5

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PROJECT_BIBLE.md                          │
│                    (Canonical Governance)                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┬─────────────────────┐
        │                           │                       │
   ┌────▼────┐              ┌──────▼──────┐        ┌──────▼──────┐
   │  Engine  │              │Risk Manager │        │  Telemetry  │
   │  Core    │◄────────────►│   Antifragile│◄──────►│  Prometheus │
   └────┬────┘              └──────┬──────┘        └──────┬──────┘
        │                           │                       │
   ┌────▼────────────┬──────────────┴───────────┬─────────▼──────┐
   │   Arbitrage     │      Flash Loans         │   Liquidation   │
   │   Strategy      │      Strategy            │   Strategy      │
   └─────────────────┴──────────────────────────┴─────────────────┘
                                    │
                           ┌────────▼────────┐
                           │  Smart Contracts│
                           │  (MEVExecutor)  │
                           └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- 10+ ETH for mainnet operations

### Installation

```bash
# Clone repository
git clone https://github.com/mev-og/mev-v3
cd mev-v3

# Verify compliance
make compliance

# Install dependencies
make install

# Copy environment template
cp .env.example .env
# Edit .env with your configuration

# Run tests
make test

# Start development environment
make dev
```

### Production Deployment

```bash
# Build production images
make build

# Deploy to Kubernetes
make deploy

# Verify deployment
kubectl get pods -n mev-v3
```

## 📊 Monitoring

Access monitoring dashboards:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9091
- Agent API: http://localhost:8081

## 🧪 Testing

The system enforces comprehensive testing gates:

```bash
# Run all tests
make test

# Run specific test suites
python -m pytest tests/test_compliance.py -v  # Compliance tests
python -m pytest tests/test_strategies.py -v  # Strategy tests
npx hardhat test                              # Smart contract tests
forge test -vvv                               # Foundry tests

# Run simulation
make simulation

# Run chaos testing
make chaos
```

## 🔧 Configuration

Key configuration files:
- `PROJECT_BIBLE.md` - Canonical governance rules (DO NOT MODIFY)
- `config.json` - System configuration
- `.env` - Environment variables
- `k8s/` - Kubernetes manifests

## 📈 Strategies

### 1. Arbitrage
- Multi-DEX price discrepancy detection
- Optimal routing with slippage protection
- MEV-resistant execution via Flashbots

### 2. Flash Loans
- Multi-provider support (Aave, Balancer, dYdX)
- Capital-efficient arbitrage and liquidations
- Atomic transaction bundles

### 3. Liquidations
- Multi-protocol monitoring (Aave, Compound, Maker)
- Health factor tracking
- Priority-based execution

## 🛡️ Risk Management

Antifragile risk system with:
- Dynamic position sizing (Kelly Criterion)
- Circuit breakers
- Correlation risk monitoring
- Real-time VaR/CVaR calculation

## 🔄 Mutation System

Strategies self-mutate based on performance:
- Parameter optimization
- Strategy pruning
- Threshold adjustments

## 🚨 DRP & Chaos Testing

Weekly disaster recovery drills per PROJECT_BIBLE:
- Service failures
- Network partitions
- Resource exhaustion
- Key loss scenarios

## 🤖 Agent API

RESTful API for LLM integration:

```bash
# Register agent
curl -X POST http://localhost:8081/agent/register \
  -d "agent_id=my-agent&role=analyst"

# Query metrics
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8081/metrics

# Execute commands
curl -X POST http://localhost:8081/system/command \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"command": "analyze", "agent_id": "my-agent"}'
```

## 📝 Compliance

All components must maintain:
- `project_bible_compliant: true`
- `mutation_ready: true`
- Required metadata in all files
- Passing test gates

Check compliance:
```bash
make compliance
```

## 🔐 Security

- No hardcoded secrets (use GCP Secret Manager)
- All keys stored in environment variables
- Smart contract audits via Slither
- Regular security scans

## 📚 Documentation

- [PROJECT_BIBLE.md](PROJECT_BIBLE.md) - Canonical rules
- [AGENTS.md](AGENTS.md) - Agent integration
- [API Documentation](docs/api.md)
- [Strategy Guide](docs/strategies.md)

## ⚠️ Warnings

1. **NEVER** modify PROJECT_BIBLE.md without proper governance
2. **NEVER** disable compliance checks
3. **NEVER** commit secrets to the repository
4. **ALWAYS** run simulations before mainnet deployment
5. **ALWAYS** maintain thresholds above minimums

## 🤝 Contributing

1. Read PROJECT_BIBLE.md thoroughly
2. Ensure all code is mutation-ready
3. Include required metadata
4. Pass all test gates
5. Submit PR with compliance block

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 🆘 Support

- Discord: [MEV-OG Discord](https://discord.gg/mev-og)
- Issues: [GitHub Issues](https://github.com/mev-og/mev-v3/issues)
- Docs: [Documentation](https://docs.mev-og.com)

---

**Remember:** This system is designed for autonomous operation. Human intervention should be minimal and only for meta-governance. Trust the PROJECT_BIBLE.
