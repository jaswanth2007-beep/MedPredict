# MedPredict — Predictive Maintenance for Turbofan Engines

MedPredict is a **Machine Learning-based predictive maintenance system** that analyzes turbofan engine sensor data to estimate the **Remaining Useful Life (RUL)** of an engine and identify its failure risk.

The project uses historical engine sensor data to train a machine learning model and provides an interactive **Streamlit dashboard** for monitoring engine health and predictions.

## 🚀 Features

* Predicts **Remaining Useful Life (RUL)** of turbofan engines
* Uses multiple engine sensor readings for prediction
* Classifies engine condition into **Low, Medium, High, and Critical risk**
* Visualizes sensor trends and engine health
* Interactive Streamlit web dashboard
* Machine Learning model based on **XGBoost**
* Data preprocessing and feature engineering pipeline

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib**
* **Streamlit**

## 📊 Dataset

The project uses the **NASA C-MAPSS turbofan engine dataset**, which contains simulated run-to-failure sensor data from aircraft engines.

The dataset is used to learn patterns in engine degradation and estimate how many operating cycles remain before failure.

## ⚙️ How It Works

```text
Engine Sensor Data
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
XGBoost ML Model
        ↓
RUL Prediction
        ↓
Risk Assessment
        ↓
Streamlit Dashboard
```

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/MedPredict_GE_EID_Project.git
cd MedPredict_GE_EID_Project
```

Create and activate a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📈 Output

The system provides:

* Predicted **Remaining Useful Life**
* Engine health status
* Failure risk level
* Sensor readings and trends
* Interactive visualizations

## 🎯 Objective

The main objective of MedPredict is to demonstrate how **Machine Learning can be used for predictive maintenance**, allowing potential engine failures to be identified earlier and helping reduce unexpected downtime and maintenance costs.

## 🔮 Future Improvements

* Real-time sensor data integration
* More advanced deep learning models such as LSTM
* Cloud deployment
* Automated maintenance alerts
* Integration with IoT sensor systems
* Improved model performance and explainability

## 👨‍💻 Author

**Borra Jaswanth Reddy**

---

⭐ If you find this project useful, consider giving the repository a star!
