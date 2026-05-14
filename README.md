# Constrained CartPole RL Benchmarking Suite

![CartPole](https://img.shields.io/badge/Environment-Gymnasium-green)
![RL-Algos](https://img.shields.io/badge/Algorithms-DQN%20|%20Q--Learning%20|%20SARSA-blue)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)

This repository is a robust framework for training, tuning, and stress-testing **Reinforcement Learning** agents on a constrained CartPole-v1 environment. Designed for portability between Google Colab and local environments, it includes automated data logging and physical robustness analysis.

---

## 🚀 Key Features

*   **Multi-Algorithm Support:** Switch between **Deep Q-Networks (DQN)**, **Q-Learning**, and **SARSA** using a unified interface.
*   **Interactive Training GUI:** Uses `ipywidgets` to select state modes (2D vs 4D), action types, and stopping logic (Fixed episodes vs. Overfitting detection) without touching the code.
*   **Physical Stress Testing:** Evaluates model resilience by simulating hardware/environmental shifts:
    *   **Initial Pole Angle:** Tests recovery from extreme tilt (up to 14°).
    *   **Pole Mass:** Tests how weight variations (0.1kg to 2.0kg) affect balance stability.
*   **Automated Workflow:** Built-in logic for hyperparameter tuning, model saving via `joblib`, and automated GitHub commits directly from Colab.
*   **Visualization:** Automatically renders training videos and generates Seaborn-powered performance graphs for stress analysis.

---

## 📁 Repository Structure

*   `Code/`: Core Python modules including agents, environment wrappers, and tuning logic.
*   `Best One/`: Stores the "Golden" models—the highest performing `.pkl` files and logs.
*   `Results/`: (Git ignored) Temporary directory for training logs and session videos.
*   `requirements.txt`: List of dependencies (Gymnasium, Torch, Joblib, etc.).

---

## 💻 Setup & Usage

### Running on Google Colab
The project is optimized for a Colab-first workflow. The setup cells in the notebook will:
1. Clone the repository.
2. Install system dependencies (`xvfb`, `ffmpeg`) for video rendering.
3. Install Python requirements.

### Local Execution
1. Clone the repo: `git clone https://github.com/SaxenaNamanBase/constrained-cartpole-rl.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure you have a virtual display configured if running on a headless Linux server.

---

## 📊 Evaluation & Analysis

The **Stress Test** suite is the core of this project’s analytical capability. It doesn't just check if the agent works; it identifies the **Critical Zone**—the exact physical threshold where an algorithm's reliability begins to fail.

*   **Angle Stress:** Measures the "Basin of Attraction" for the controller.
*   **Mass Stress:** Analyzes how the controller handles changes in inertia and gravity.

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
