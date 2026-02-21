# Optimizing Assembly Line Efficiency Using an RL-Powered Digital Twin Framework

**Author:** Aswin Balaji R  
**Academic Program:** M.Tech. Artificial Intelligence & Machine Learning, BITS Pilani (WILP)  
**Domain:** Artificial Intelligence, Production & Operations Management

---

## Project Overview

The manufacturing industry relies heavily on static, rule-based heuristics to manage production scheduling. When faced with sudden demand spikes or stochastic machine failures, these rigid systems often fail, leading to massive operational waste (excessive overtime) or missed deadlines. 

This project tackles this problem by integrating a Discrete-Event Simulation (DES) with Deep Reinforcement Learning (DRL). By building a high-fidelity Digital Twin prototype of a three-station assembly line featuring stochastic failures and complex shift constraints, an autonomous Proximal Policy Optimization (PPO) agent was trained to handle complex, multi-discrete decision-making. The agent learns to manage part sequencing, station halting, and overtime activation without explicit manual programming.

### Key Achievement: The "High Priority Rush" Stress Test
To prove the system's intelligence, it was subjected to a mathematically impossible load where order volume vastly exceeded the bottleneck's theoretical capacity.

* **Traditional "Smart" Manager Heuristic:** Followed rigid rules, panicked, and triggered 73.0 hours of overtime, achieving an 11.57% On-Time Delivery (OTD) rate.
* **The RL Agent:** Anticipated the bottleneck physics, realized overtime would not physically clear the queue, and utilized only 0.2 hours of overtime while achieving a slightly better 11.63% OTD rate. 
* **Result:** The RL agent eliminated over 72 hours of operational waste while matching delivery targets by utilizing predictive intelligence rather than reactive rules.

---

## System Architecture

The framework architecture intentionally decouples the physical process model from the decision-making intelligence:

* **Digital Twin Prototype (Environment):** Built using the SimPy library to model the physical physics, a 22-28 minute bottleneck at Station 2, an 8% machine failure rate, and dynamic order book logic.
* **Intelligence Layer (RL Agent):** Powered by the Stable-Baselines3 implementation of the Proximal Policy Optimization (PPO) algorithm, utilizing a Multi-Layer Perceptron (MLP) policy.
* **Interface:** A Gymnasium interface wrapper translates raw simulation data into a 35-dimensional state vector and normalizes it for the agent.

---

## Core Innovations

### 1. Temporal Awareness via Cyclical Time Features
Traditional algorithms are time-blind. To provide context, human shift schedules (08:00–18:30) and lunch breaks were mathematically encoded using the Sine and Cosine transformations of the time-of-day. This allowed the agent to map time onto a unit circle, anticipating discrete operational boundaries and sequencing parts to avoid off-shift overtime.

### 2. Multi-Objective Reward Engineering (Solving the Lazy Agent)
To enforce Lean Manufacturing principles, the reward function was scalarized to balance conflicting Key Performance Indicators (KPIs):
* **+50.0** High Priority Completion Reward
* **-60.0** High Priority Lateness Penalty (Prevents the "lazy agent" from halting production entirely to save costs)
* **-0.75** Overtime Usage Penalty
* **-0.20** Work-In-Progress (WIP) Penalty (Discourages buffer hoarding and drives the strategic "Halt" action during machine failures)

### 3. High Predictive Accuracy
The agent was trained for 5,500,000 timesteps using Domain Randomization. The Value Function Explained Variance consistently reached levels above 0.97, proving the agent achieved a near-perfect understanding of the assembly line's underlying physical dynamics.

---

## Repository Structure

* `simulation_model.py`: Defines the AssemblyLineSim class, which serves as the core physics engine, tracking order books and repair loops.
* `assembly_line_env.py`: The Gymnasium wrapper responsible for state normalization, cyclical temporal feature extraction, and action decoding.
* `train_and_evaluate.py`: Contains the logic for the decision layer, executing the PPO agent and baseline models.
* `dashboard.py`: User interface and control layer for live scenario configuration and metric tracking.

---

*Disclaimer: This repository contains logic developed for an academic dissertation at BITS Pilani. To comply with standard confidentiality practices, specific processing times, failure rates, and parameters are generalized and do not reflect the exact proprietary metrics of any specific commercial manufacturing facility.*
