# Optimizer

This folder contains the reusable optimization engine.

Currently implemented:
- `qisa.py`: Quantum-Inspired Simulated Annealing (QISA)

The optimizer is problem-agnostic; scheduling modules provide:
- state representation
- neighbor generator
- energy/fitness function (hard constraints are heavily penalized)
