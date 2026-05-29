# Constrained CartPole RL Benchmarking Suite

![CartPole](https://img.shields.io/badge/Environment-Gymnasium-green)
![RL-Algos](https://img.shields.io/badge/Algorithms-DQN%20|%20Q--Learning%20|%20SARSA-blue)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)

This repository is a robust framework for training, tuning, and stress-testing **Reinforcement Learning** agents on a constrained CartPole-v1 environment. Designed for portability between Google Colab and local environments, it includes automated data logging and physical robustness analysis.

---

## 🚀 Key Features

*   **Multi-Algorithm Support:** Switch between **Deep Q-Networks (DQN)**, **Q-Learning**, and **SARSA** using a unified interface.
*   * **Custom Environment Wrapper (`GymWrapper`):** A custom-built wrapper crucial for managing environment constraints. It dynamically filters states between **2D (Pole Only)** and **4D (Full State Physics)** and handles action space translations—enabling continuous action space mapping specifically tailored for the DQN agent
*   **Interactive Training GUI:** Uses `ipywidgets` to select state modes (2D vs 4D), action types, and stopping logic (Fixed episodes vs. Overfitting detection) without touching the code.
*   **Physical Stress Testing:** Evaluates model resilience by simulating hardware/environmental shifts:
    *   **Initial Pole Angle:** Tests recovery from extreme tilt (up to 14°).
    *   **Pole Mass:** Tests how weight variations (0.1kg to 2.0kg) affect balance stability.
*   **Automated Workflow:** Built-in logic for hyperparameter tuning, model saving via `joblib`, and automated GitHub commits directly from Colab.
*   **Visualization:** Automatically renders training videos and generates Seaborn-powered performance graphs for stress analysis.

---

## 🗺️ Project Roadmap

* ✅ **Multi-algorithm core setup** (DQN, Q-Learning, SARSA)
* ✅ **Custom `GymWrapper` integration** for 2D/4D states and continuous datasets
* ✅ **Interactive `ipywidgets` deployment UI dashboard**
* ✅ **Automated physical stress-testing metrics** (Angle/Mass constraints)
* 🔲 **Physical deployment translation validation** across hardware interfaces

## 📁 Repository Structure

```text
├── Code/                         # Core Python application files
│   ├── CartPoleTest.ipynb        # Main notebook to act as an interface for the model
│   ├── cartPoleSetup.ino         # Directions for hardware interface based on Arduino.
│   ├── config.py                 # Centralized hyperparameter and environment configurations
│   ├── control_algorithm.py      # Implementation classes for Q-Learning, SARSA, and DQN agents
│   ├── cross_validation.py       # Cross-validation loop utilities and model training drivers
│   ├── data_logger.py            # Automated tracking for rewards, iterations, and performance logs
│   ├── exploration_strategies.py # Epsilon-Greedy and Softmax strategy modules
│   ├── gym_wrapper.py            # Custom wrapper handling state reduction and continuous actions
│   ├── hardware_interface.py     # Setup infrastructure for physical testbed deployment
│   ├── hyperparameter_tuning.py  # Optimization functions for automated parameter search
├── Results/                      # (Git ignored) Folder for training logs, plots & videos for each iteration
├── requirements.txt              # Full installation dependency manifest
└── README.md                     # Documentation
```

##💻 Setup & Usage
###⚙️ Prerequisites
All primary environment requirements are isolated within requirements.txt. Core dependencies rely on gymnasium, torch, joblib, and ipywidgets.

###🌐 Google Colab Workflow

The project is optimized for a Colab-first workflow. The setup cells in the notebook will:
1. Clone the repository.
2. Install system dependencies (`xvfb`, `ffmpeg`) for video rendering.
3. Install Python requirements.

This repository is optimized for an ordered, notebook-first training approach. Open your .ipynb notebook file and execute the cells sequentially through these architectural sections:

Necessary Setup: Automatic environment detection, repository cloning, system virtual display (xvfb/ffmpeg) instantiation, and dependency handling.

Importing Files & Credentials: Handles secure connections to Google Drive and parses encrypted GitHub Tokens (GH_TOKEN) for direct commits.

Functions and Code Import: Loads the foundational modular code blocks from the Code/ directory into the active kernel memory space.

Training the Models: Renders the interactive widget dashboard. Configure your selections and click "Launch Training" to execute optimization cycles.

Testing: Runs validation episodes against the trained agent configurations and renders browser-playable playback videos of peak evaluation runs.

Stress Test: Injects environmental physical stressors to determine edge-case balance failure limitations.

### Local Execution
1. Clone the repo: `git clone https://github.com/SaxenaNamanBase/constrained-cartpole-rl.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure you have a virtual display configured if running on a headless Linux server.

---

##📊 Evaluation & Analysis

The framework utilizes a multi-tier assessment methodology to validate how effectively models stabilize the system under shifting constraints.

1. **Standard Training & Testing Validation**
Performance metrics capture standard reward accretion plots across epochs. Evaluation routines automatically save performance logs locally to testing_performance_logs.csv and export video tracking arrays to identify behavioral optimizations.


2. **Physical Stress Testing**
The stress suite aggressively updates internal physics arrays to map out model stability envelopes and determine the Critical Zone (the specific boundary point where stability degrades).

Angle Distribution: Discovers the system's "Basin of Attraction" boundaries when managing intense starting tilt degrees.

Mass Variations: Gauges mathematical model durability against changing momentum vectors and structural weight discrepancies.

---

## 🧪 Requirements

*   `gymnasium`
*   `torch`
*   `joblib`
*   `matplotlib` & `seaborn`
*   `ipywidgets`
*   `pyvirtualdisplay`

---

## 📝 Author's Note
This project was developed as a bridge between RL simulation and physical testbeds. Use the **DQN** implementation for the best results in high-dimensional (4D) state spaces.
