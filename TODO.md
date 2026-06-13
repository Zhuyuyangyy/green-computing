# TODO.md - Innovation Suggestions & Technical Debt

## Innovation Suggestions

### 1. Data Center Carbon Footprint Tracking System

**Priority:** HIGH
**Estimated Effort:** 3-4 weeks
**Impact:** High - enables real-time carbon accounting

Implement a comprehensive carbon footprint tracking module that:

- Integrates with real-time carbon intensity APIs (WattTime, ElectricityMaps, Carbon Interface)
- Tracks Scope 1 (direct), Scope 2 (electricity), and Scope 3 (supply chain) emissions
- Provides per-workload carbon attribution (carbon cost per training job, per inference request)
- Generates carbon reports aligned with GHG Protocol standards
- Supports carbon offset integration and renewable energy certificate (REC) tracking

**Technical Approach:**
- Extend `DataCenterEnv` with a `CarbonTracker` module
- Add API clients for carbon intensity data sources
- Implement workload-level carbon attribution using power proportionality
- Create reporting dashboard with historical trends

---

### 2. Carbon-Aware Task Scheduling

**Priority:** HIGH
**Estimated Effort:** 4-6 weeks
**Impact:** High - directly reduces carbon emissions by 20-40%

Implement carbon-aware task scheduling that defers flexible workloads to periods of low carbon intensity:

- Batch job scheduling with carbon awareness (training jobs, data processing)
- Geographic workload routing to regions with cleaner grids
- SLA-aware scheduling that respects latency constraints
- Integration with Kubernetes for production deployment

**Technical Approach:**
- Extend hour-level SAC agent with scheduling actions
- Add workload flexibility classification (rigid vs. deferrable)
- Implement carbon-aware priority queues
- Add geographic carbon intensity comparison module

---

### 3. Renewable Energy Matching & Forecasting

**Priority:** HIGH
**Estimated Effort:** 4-6 weeks
**Impact:** High - maximizes renewable energy utilization

Implement renewable energy matching that aligns data center consumption with renewable generation:

- Solar/wind generation forecasting using weather data
- Battery storage optimization for renewable smoothing
- Power Purchase Agreement (PPA) simulation
- 24/7 clean energy matching (not just annual offsets)

**Technical Approach:**
- Add renewable energy forecasting model (LSTM/Transformer)
- Extend environment with battery storage simulation
- Implement PPA pricing model
- Add 24/7 clean energy matching algorithm

---

### 4. PUE Prediction & Optimization

**Priority:** MEDIUM
**Estimated Effort:** 2-3 weeks
**Impact:** Medium - enables proactive cooling optimization

Implement PUE prediction using machine learning:

- Real-time PUE prediction based on workload, weather, and cooling state
- Anomaly detection for PUE degradation
- Cooling system optimization recommendations
- Integration with building management systems (BMS)

**Technical Approach:**
- Train PUE prediction model on historical data
- Add anomaly detection for PUE outliers
- Implement cooling optimization using model predictive control (MPC)
- Add BMS integration interface

---

### 5. Multi-Datacenter Coordination

**Priority:** MEDIUM
**Estimated Effort:** 6-8 weeks
**Impact:** High - enables global carbon optimization

Extend the framework to coordinate across multiple data centers:

- Geographic workload routing based on carbon intensity
- Inter-datacenter load balancing
- Global carbon budget management
- Disaster recovery with carbon awareness

**Technical Approach:**
- Create `MultiDataCenterEnv` with multiple facility models
- Implement geographic routing agent
- Add inter-datacenter communication protocol
- Extend reward function for global carbon optimization

---

### 6. Model-Based RL for Sample Efficiency

**Priority:** MEDIUM
**Estimated Effort:** 4-6 weeks
**Impact:** Medium - reduces training time by 10x

Implement model-based RL to improve sample efficiency:

- World model for data center dynamics
- Dreamer-style imagination-based training
- Model predictive control (MPC) for planning
- Uncertainty-aware planning for safety

**Technical Approach:**
- Train dynamics model on environment transitions
- Implement imagination rollouts for policy optimization
- Add uncertainty quantification for safety constraints
- Compare sample efficiency with current model-free approach

---

### 7. Real-Time Telemetry Integration

**Priority:** MEDIUM
**Estimated Effort:** 3-4 weeks
**Impact:** Medium - enables production deployment

Integrate with real data center monitoring systems:

- Prometheus metrics ingestion
- Grafana dashboard integration
- SNMP/IPMI for hardware telemetry
- Real-time anomaly detection

**Technical Approach:**
- Add Prometheus client for metrics export
- Create Grafana dashboard templates
- Implement hardware telemetry adapters
- Add real-time anomaly detection pipeline

---

## Technical Debt

### High Priority

1. **Add `step()` method to DataCenterEnv** - The `evaluate_mode()` in main.py previously called `env.step()` which doesn't exist. Fixed to use `step_second()` but a unified step interface would be cleaner.

2. **Standardize action space** - Different levels use different action spaces (normalized [0,1] vs [-1,1]). Standardize to a common interface.

3. **Add proper Gym API compliance** - Register as a proper Gym environment for compatibility with standard RL libraries.

### Medium Priority

4. **Remove hardcoded paths in scripts** - `scripts/run_training.sh` contains `/mnt/d/` hardcoded paths.

5. **Add configuration management** - Move hardcoded hyperparameters to YAML config files.

6. **Improve error handling** - Add proper error handling for missing dependencies, invalid configurations, etc.

7. **Add logging framework** - Replace print statements with proper Python logging.

### Low Priority

8. **Type hints completion** - Add complete type hints to all functions.

9. **Docstring standardization** - Ensure all public methods have complete docstrings.

10. **Code formatting** - Run ruff formatter on all source files.

---

## Feature Requests

- [ ] Support for AMD GPU power models
- [ ] Liquid cooling simulation
- [ ] Free cooling (economizer) modeling
- [ ] Waste heat recovery simulation
- [ ] Carbon offset marketplace integration
- [ ] Multi-tenant carbon attribution
- [ ] Edge computing carbon optimization
- [ ] Quantum computing cooling models
