# joy-tech_labs_intership
# 🚀 Anomaly Detection on NASA SMAP/MSL Telemetry Dataset

An end-to-end anomaly detection project built during my Machine Learning internship using the NASA SMAP/MSL spacecraft telemetry dataset.

The project focuses on detecting anomalies in multivariate time-series sensor data using machine learning techniques. Currently, the repository contains a complete implementation of **Isolation Forest**, with future work including **LSTM-based** and **Autoencoder-based** anomaly detection models.

---

## 📌 Project Objectives

- Explore and understand the NASA SMAP/MSL telemetry dataset
- Perform exploratory data analysis (EDA)
- Build an Isolation Forest anomaly detection pipeline
- Evaluate model performance across multiple telemetry channels
- Compare different contamination strategies
- Extend the project with deep learning models (LSTM & Autoencoder)

---

## 📂 Repository Structure

```
.                  
├── notebooks/             
│   ├── 01_eda.py
│   └── 02_isolation_forest.py
|   └── 03_hyperparameter_experiment.py
├── src/                   # Source code
│   ├── data/
|   |   └── loader.py
|   |   └── preprocessing.py
|   |   └── windowing.py
│   ├── models/
|   |   └── base_detector.py
|   |   └── isolation_forest.py
│   ├── evaluation/
|   |   └── eda_utils.py
|   |   └── metrics.py
│   └── utils/
|   |   └── config.py
|   |   └── visualization.py
├── results/
│   ├── contamination_experiment_summary.csv
│   ├── isolation_forest_per_channel_results.csv
│   ├── if_per_channel_fixed_low_0.05.csv
│   ├── if_per_channel_fixed_moderate_0.15.csv
│   └── if_per_channel_oracle_leaky.csv
├── models/
│   └── isoforest_all_channels.joblib
└── README.md
```

---

## 📊 Dataset

This project uses the **NASA SMAP/MSL Telemetry Dataset**, a benchmark dataset widely used for time-series anomaly detection research.

The dataset contains:

- Multiple telemetry channels
- Training data containing normal behaviour
- Testing data containing anomalies
- Ground truth anomaly intervals

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Joblib

---

## 🔍 Current Implementation

### ✅ Exploratory Data Analysis (EDA)

- Dataset exploration
- Channel-wise visualization
- Distribution analysis
- Anomaly inspection
- Missing value checking

### ✅ Isolation Forest

Implemented:

- Per-channel Isolation Forest
- Fixed contamination (0.05)
- Fixed contamination (0.15)
- Oracle contamination (for comparison)
- Model serialization using Joblib

Generated outputs:

- Channel-wise predictions
- Evaluation metrics
- Contamination comparison
- Summary reports

---

## 📈 Evaluation

The repository includes experiments comparing different contamination values and their effect on anomaly detection performance.

Evaluation includes:

- Precision
- Recall
- F1-score
- Channel-wise analysis

---

## 📁 Generated Files

Some important outputs include:

- `isoforest_all_channels.joblib`
- `contamination_experiment_summary.csv`
- `isolation_forest_per_channel_results.csv`
- `if_per_channel_fixed_low_0.05.csv`
- `if_per_channel_fixed_moderate_0.15.csv`
- `if_per_channel_oracle_leaky.csv`

---

## 🚧 Future Work

- Implement LSTM-based anomaly detection
- Implement Autoencoder-based anomaly detection
- Compare all three approaches
- Hyperparameter tuning
- Interactive visualizations
- Performance benchmarking

---

## ▶️ Getting Started

### Clone the repository

```bash
git clone https://github.com/vidhimishra2007/joy-tech_labs_intership.git
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run EDA

```bash
python project/notebooks/01_eda.py
```

### Run Isolation Forest

```bash
python project/notebooks/02_isolation_forest.py
```

---

## 📚 Learning Outcomes

Through this project I learned:

- Time-series anomaly detection
- Isolation Forest algorithm
- Feature engineering
- Model evaluation
- Experiment tracking
- Working with real-world telemetry data
- Structuring ML projects for production

---

## 👩‍💻 Author

**Vidhi Mishra**

Machine Learning Intern | AI & Data Science Enthusiast

GitHub: https://github.com/vidhimishra2007

---

## ⭐ If you found this repository useful, consider giving it a star!
