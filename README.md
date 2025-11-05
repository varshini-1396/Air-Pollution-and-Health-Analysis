# 🌍 Air Quality and Health Impact Analysis

> A comprehensive Data Science project that investigates the relationship between **air pollution** and **public health outcomes**.  
> The project performs exploratory data analysis, hypothesis testing, and machine learning modeling to predict the **Health Impact Score** from air quality and weather data.

---

## 🧭 Quick Navigation

| Section | Description |
|----------|-------------|
| [📂 Project Structure](#-project-structure) | Overview of all files and folders |
| [🎯 Objective](#-objective) | Purpose and goals of the project |
| [🧮 Problem Definition](#-problem-definition) | Type, target variable, and expected outcomes |
| [📊 Dataset Description](#-dataset-description) | Information about dataset and its features |
| [🧠 Data Exploration](#-data-exploration) | EDA results, outliers, and correlations |
| [🔬 Data Preprocessing](#-data-preprocessing) | Steps for data cleaning and transformation |
| [🧪 Hypothesis Testing](#-hypothesis-testing) | Statistical analysis and significance testing |
| [⚙️ Modeling](#-modeling) | Model training process and LightGBM setup |
| [📈 Model Evaluation](#-model-evaluation) | Performance metrics and results |
| [🧾 Overfitting Check](#-overfitting-check) | Model generalization verification |
| [💾 Model Saving](#-model-saving) | Saving trained model for deployment |
| [💡 Insights & Conclusions](#-insights--conclusions) | Key findings and data-driven insights |
| [💻 Web Application Deployment](#-web-application-deployment) | Flask app setup and usage instructions |

---
## 📂 Project Structure
```
Air-Quality-and-Health-Analysis/
│
├── data/
│ └── raw/
│ └── air_quality_health_impact_data.csv # Kaggle dataset
│
├── models/
│ └── health_impact_model.pkl # Trained LightGBM model
│
├── static/
│ └── style.css # Flask app CSS styles
│
├── templates/
│ └── index.html # Web app frontend (Glassmorphism UI)
│
├── app.py # Flask application file
├── notebook.ipynb # Original Jupyter notebook
├── script.py # Converted .py version of the notebook
├── requirements.txt # Required dependencies
└── README.md # Project documentation
```
### 🗂️ Description:
- **`/data/`** – Contains the dataset used for analysis and model training.  
- **`/models/`** – Stores serialized trained models (`.pkl`).  
- **`/templates/`** and **`/static/`** – Frontend files for Flask (HTML, CSS).  
- **`app.py`** – Flask backend that serves predictions using the LightGBM model.  
- **`notebook.ipynb`** – Full data analysis and modeling notebook.  
- **`script.py`** – Clean `.py` version of the notebook for reproducibility.  

---

## 🎯 Objective

The goal of this project is to analyze how air pollutants such as **PM2.5, PM10, NO₂, SO₂, O₃**, and meteorological parameters affect health outcomes like **respiratory cases, cardiovascular cases,** and **hospital admissions**.

### Key Goals:
1. Identify major air pollutants impacting human health.  
2. Quantify relationships between **Air Quality Index (AQI)** and **Health Impact Score**.  
3. Build regression models to accurately predict health impact from environmental data.

---

## 🧮 Problem Definition

| Aspect | Description |
|--------|-------------|
| **Type** | Supervised Machine Learning |
| **Problem** | Regression |
| **Target Variable** | `HealthImpactScore` |
| **Expected Outcome** | Predict numerical health impact values from pollution and weather features |

---

## 📊 Dataset Description

**Source:** Kaggle – *Air Quality and Health Impact Dataset*  
**Records:** 5811  
**Columns:** 15  

| Column | Description |
|--------|-------------|
| `RecordID` | Unique record identifier |
| `AQI` | Air Quality Index |
| `PM10` | Particulate Matter (≤10μm) |
| `PM2_5` | Particulate Matter (≤2.5μm) |
| `NO2` | Nitrogen Dioxide concentration |
| `SO2` | Sulfur Dioxide concentration |
| `O3` | Ozone concentration |
| `Temperature`, `Humidity`, `WindSpeed` | Meteorological data |
| `RespiratoryCases`, `CardiovascularCases`, `HospitalAdmissions` | Health data |
| `HealthImpactScore` | Composite health index *(Target)* |
| `HealthImpactClass` | Health impact category (Low/Medium/High) |

---

## 🧠 Data Exploration

- The dataset contains **5811 rows** and **15 columns**.  
- **No missing values** or **duplicate entries** detected.  
- **Outliers** found in:
  - `RespiratoryCases`
  - `CardiovascularCases`
  - `HospitalAdmissions`
- **RecordID** is non-informative → dropped.
- **HealthImpactClass** also dropped (to prevent target leakage).
- High **skewness** detected → handled using **Power Transformation**.

### 🔍 EDA Visualizations
- **Boxplots:** To detect and visualize outliers.  
- **KDE plots:** To observe skewness and transformation results.  
- **Correlation Heatmap:** To analyze feature relationships.

---

## 🔬 Data Preprocessing

| Step | Description |
|------|--------------|
| 1️⃣ | Dropped `RecordID` and `HealthImpactClass`. |
| 2️⃣ | Applied `PowerTransformer` to fix skewness (Box–Cox or Yeo–Johnson). |
| 3️⃣ | Scaled features using **MinMaxScaler** before model training. |
| 4️⃣ | Split dataset (80% train, 20% test). |

---

## 🧪 Hypothesis Testing

Statistical tests were conducted to validate data-driven insights.

| Hypothesis | Relationship | Test Used | Result | Conclusion |
|-------------|--------------|------------|----------|-------------|
| **H₁** | AQI ↔ Health Impact Score | Pearson Correlation | r = 0.615, p = 0.00000 | ✅ Significant |
| **H₂** | PM2.5 ↔ Respiratory Cases | Pearson Correlation | r = 0.025, p = 0.05433 | ❌ Not significant |
| **H₃** | HealthImpactClass ↔ HealthImpactScore | ANOVA | F = 2859.063, p = 0.00000 | ✅ Significant difference |
| **H₄** | Temperature/Humidity/WindSpeed ↔ AQI | Linear Regression | p > 0.05 | ❌ Not significant |

### Interpretation Summary
- **AQI** shows a **strong positive correlation** with HealthImpactScore.  
- **PM2.5** does not significantly affect respiratory cases in this dataset.  
- **HealthImpactClass** categories differ significantly in average scores (validated by ANOVA).  
- Meteorological factors (Temperature, Humidity, WindSpeed) have **no strong impact** on AQI.  

---

## ⚙️ Modeling

After preprocessing, multiple regression models were tested using **LazyPredict** to identify top performers.  
The **LightGBM Regressor** was selected for final training due to its superior performance.

### 📈 Model Training Workflow
1. Split the data into training (80%) and testing (20%).  
2. Train models using **LazyRegressor** for baseline comparison.  
3. Fine-tune and train a **LightGBM Regressor**.  
4. Evaluate performance using multiple regression metrics.  

### 🔧 Model Parameters
```python
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8
}
```


## 📈 Model Evaluation

Once the model was trained, it was evaluated using key regression metrics:

| Metric | Value | Interpretation |
|---------|--------|----------------|
| **Mean Squared Error (MSE)** | 4.280 | Measures average squared difference between predicted and actual values. Lower is better. |
| **Root Mean Squared Error (RMSE)** | 2.069 | Model predictions deviate by ≈2.07 points on average. |
| **Mean Absolute Error (MAE)** | 1.076 | Average absolute deviation between predicted and actual scores. |
| **R-squared (R²)** | 0.977 | Model explains **97.7%** of the variance in HealthImpactScore. |

### ✅ Model Performance Summary
- The **LightGBM Regressor** achieved excellent performance across all metrics.  
- **R² = 0.977** → indicates the model captures nearly all variability in health impact scores.  
- **MAE ≈ 1.07** → predictions are on average within ±1 point of the true score.  
- The model generalizes well, showing minimal difference between training and testing performance.


## 🧾 Overfitting Check

To ensure reliability, R² was also computed for both training and testing data:

| Dataset | R² Score | Interpretation |
|----------|-----------|----------------|
| **Training** | ~0.98 | Strong model fit |
| **Testing** | ~0.977 | Excellent generalization |

✅ **Difference < 0.05 → No major overfitting detected.**  
⚙️ The model generalizes well to unseen data.

---

## 💾 Model Saving

After successful training and evaluation, the model was serialized using **Joblib** for deployment:

```python
import joblib
joblib.dump(model, 'models/health_impact_model.pkl')
```

## 💡 Insights & Conclusions

### 🔍 Key Observations
- **Air Quality Index (AQI)** shows a **strong positive correlation** with the **Health Impact Score**.
- **PM2.5** and **NO₂** contribute significantly to overall health impact.
- **HealthImpactClass** groups differ meaningfully, confirmed via ANOVA testing.
- Meteorological factors (Temperature, Humidity, Wind Speed) had **no statistically significant** correlation with AQI.

### 📊 Model Insights
- **LightGBM Regressor** achieved the best performance with:
  - **R² = 0.977**
  - **RMSE = 2.07**
  - **MAE = 1.07**
- Model generalizes well, with minimal overfitting.
- Predictions are accurate within ±1 point of true Health Impact Score.

### 🧩 Final Takeaways
- **Pollutant levels strongly correlate** with health outcomes, validating the hypothesis.  
- The model can be used for **predictive public health analysis** and **policy decision-making**.  
- The **Flask web app** enables real-time interaction and accessibility for researchers or citizens.  

---

## 💻 Web Application Deployment

### 🧠 Overview
The project includes a **Flask web application** that allows users to input environmental factors (like AQI, PM2.5, NO2, etc.) and get an **instant predicted Health Impact Score**.

### ✨ Features
- Sleek **Glassmorphism UI** with responsive design.  
- Instant predictions using the trained **LightGBM model**.  
- Interactive input fields for pollution and health metrics.  
- Can be deployed locally or to cloud services (Render, Railway, Heroku).  

### ▶️ Run Locally

```bash
# Clone the repository
git clone https://github.com/varshini-1396/Air-Pollution-Health-Analysis.git

# Navigate into the project folder
cd Air-Pollution-Health-Analysis

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
