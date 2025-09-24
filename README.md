# BEMAS: Balancing Extremes in Multi-Agent Systems

A decentralized and proximity-aware multi-agent reinforcement learning framework implementing **BEMAS (Balancing Extremes in Multi-Agent Systems)** with comparisons to IQL and PED-DQN baselines in predator–prey pursuit environments.

## Overview

BEMAS exploits the *performance spectrum* that emerges as agents learn at different rates.  
- Agents learn **from the best peers** (optimism) while avoiding **the worst peers** (pessimism).  
- A **Bayesian stability regularizer** ensures updates remain stable by limiting policy surprise.  
- Training uses **bounded local messaging**, while execution is **fully decentralized** with no communication.

This repository provides the implementation, experiments, and logging tools to reproduce results from the accompanying paper.

## Key Features

- **Optimism Shaping**: Action–value gap bonus relative to the best neighbor.  
- **Pessimism Shaping**: KL-based repulsion from the worst neighbor.  
- **Bayesian Stability**: Dirichlet belief tracking penalizes abrupt policy shifts.  
- **Phase Scheduling**: Optimism emphasized during early exploration, pessimism emphasized during later exploitation.  
- **Comparison Baselines**: Independent Q-Learning (IQL) and PED-DQN.  

## Algorithm

The reshaped reward per agent is:
\[
\tilde{r}_t^i = r_t + \alpha_t \Psi_{t,i}^{\text{opt}} + \beta_t \Gamma_{t,i} - \Lambda_{t,i}
\]

Where:
- \( \Psi \): Optimistic action–value gap (log-bounded).  
- \( \Gamma \): KL divergence repulsion from worst neighbor.  
- \( \Lambda \): Bayesian surprise penalty.  
- \( \alpha_t, \beta_t \): Scheduled weights for optimism/pessimism.

Algorithm pseudocode is provided in the paper and implemented in `agents/BEMAS/`.

## Environment Scenarios

Implemented predator–prey grid-world environments:  
- **Battery Endless**: 26 predators, 25 regenerating prey, battery resource management.  
- **Pursuit Battery**: Fixed prey, predators constrained by battery.  
- **Endless**: Coordination-focused setting without battery constraints.  

Agents operate under **partial observability**, forming **proximity-based clusters** during training.

## Installation

```bash
git clone https://github.com/Epsilon314159/BEMAS.git
cd BEMAS
pip install -r requirements.txt
