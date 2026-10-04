# Stealth Path Planner: Radar Avoidance & Routing

## What is this project?
A pathfinding engine that calculates the optimal trajectory for an aircraft to visit multiple Points of Interest (POIs) while avoiding radar detection fields. It uses search algorithms over a discretized spatial map, ensuring the detection risk never exceeds a user-defined tolerance threshold.

## Key Features
* **Electromagnetic Simulation:** Calculates maximum radar range and models the spatial detection level using 2D multivariate Gaussian attenuation, factoring in transmission power, antenna gain, and cross-section.
* **Optimal Search with A*:** Implements the A* search algorithm for navigation between POIs using the NetworkX library.
* **Custom Heuristics:** Includes Manhattan and Euclidean distance calculations, adapted with an EPSILON factor to maintain admissibility and optimize node expansion.
* **Dynamic Configuration:** Loads multiple simulation scenarios via a `scenarios.json` file, scaling from small maps to 1024x1024 grids with dozens of radars.
* **Data Visualization:** Generates heatmaps of the detection fields and plots the final step-by-step solution route using Matplotlib.

## Tech Stack
* **Language:** Python 3
* **Core Libraries:** NumPy, NetworkX
* **Visualization:** Matplotlib
* **Utilities:** tqdm, JSON

## Installation and Usage
1. Clone this repository to your local machine.
2. Install the required dependencies:
```bash
pip install numpy networkx matplotlib tqdm
