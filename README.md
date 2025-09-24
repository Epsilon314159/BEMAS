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


## Installation
~~~bash
git clone https://github.com/Epsilon314159/BEMAS.git
cd BEMAS
pip install -r requirements.txt
~~~

## Usage

### Train BEMAS
~~~bash
python main.py
~~~

### Train Baselines
~~~bash
python main.py --agent IQL --train True 
~~~

### We thank the authors of **PED-DQN** for open-sourcing their code and environment ([repo link](https://github.com/ddhostallero/PED-DQN)), which we used to reproduce PED-DQN results and ensure fair comparisons.


## Code Structure
~~~text
BEMAS/
├── agents/
│   ├── BEMAS/              # Algorithm implementation
│   ├── IQL/                # Independent Q-learning baseline
│   ├── replay_buffer.py
│   └── config_agents.py
├── envs/                   # Predator–prey grid environments
│   ├── scenarios/
│   ├── environment.py
│   ├── grid_core.py
│   └── config_env.py
├── main.py
├── config.py
└── make_env.py
~~~
