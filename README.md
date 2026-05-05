# IoT-Based Supply Chain Monitoring

This repository contains all code and data for our IoT-based supply chain monitoring project. The project uses sensor data (from Tive IoT trackers) to monitor and optimize produce transportation — analyzing temperature, weight, and shelf life to improve supply chain decisions.

---

## 📁 Repository Structure

```
IoT_Supply/
├── Code_files/
│   ├── Adaptve_.ipynb
│   ├── Combined.ipynb
│   ├── DRO (1).ipynb
│   ├── Early_RL.ipynb
│   ├── OPTIM_new.ipynb
│   ├── RO (1).ipynb
│   └── Stochastic_Programming.ipynb
│
└── Data_files/
    ├── produce_data.csv
    ├── stop_pairs_pattern.csv
    └── Tive-Report-*.csv  (sensor reference samples)
```

---

## 🗂️ Code Files

| File | Description |
|---|---|
| `Adaptve_.ipynb` | Adaptive optimization model for dynamic supply chain decisions |
| `Combined.ipynb` | Combined pipeline integrating data processing with optimization |
| `DRO (1).ipynb` | Distributionally Robust Optimization (DRO) model to handle uncertainty in supply chain parameters |
| `Early_RL.ipynb` | Early-stage Reinforcement Learning experiments for supply chain decision-making |
| `OPTIM_new.ipynb` | Updated/improved optimization model — latest version of the core optimizer |
| `RO (1).ipynb` | Robust Optimization model accounting for worst-case scenarios |
| `Stochastic_Programming.ipynb` | Stochastic programming approach for probabilistic supply chain modeling |

---

## 📊 Data Files

| File | Description |
|---|---|
| `produce_data.csv` | Synthetic/processed dataset with produce type, weight (kg), shelf life (hours), and transport temperature (°C) — used as the primary input for optimization models |
| `stop_pairs_pattern.csv` | Transport route stop-pair patterns extracted from logistics data |
| `Tive-Report-*.csv` | Raw sensor export samples from Tive IoT trackers — included as reference to show the original format of sensor readings (temperature, location, timestamps) |

---

## 🔬 Project Overview

The project explores multiple optimization strategies for cold-chain logistics:

- **Robust & Stochastic Optimization** — handles uncertainty in produce shelf life and transport conditions
- **Distributionally Robust Optimization (DRO)** — minimizes risk under ambiguous probability distributions
- **Reinforcement Learning** — trains agents to make adaptive routing/storage decisions
- **Combined Pipeline** — end-to-end integration of sensor data ingestion and optimization

---

## 🌡️ Data Source

Raw data was collected from **Tive IoT sensors** deployed during produce transportation. The sensors recorded temperature, humidity, light exposure, and GPS location at regular intervals. Sample Tive reports are included in `Data_files/` for reference.

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/chirag300/IoT_Supply.git
   ```
2. Open any notebook in `Code_files/` using Jupyter or Google Colab
3. Ensure `produce_data.csv` and `stop_pairs_pattern.csv` are accessible (update paths if needed)

---

## 📬 Contact

For questions about this project, feel free to open an issue or reach out via GitHub.
