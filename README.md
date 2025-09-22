## [ Link to interactive dashboard: https://kundanpdl-stock-prediction-main-ibbida.streamlit.app/ ]
# 📊 Stock Price Prediction Application

A machine learning project built with Streamlit for interactive data visualization and model predictions. The dataset is dynamically pulled from yahoo finance for different tickers. For the machine learning model itself, Prophet is used which is a
forecasting model. [ More info here https://facebook.github.io/prophet/ ]

## 🧰 Dependencies

* streamlit
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* prophet
* yfinance
* plotly

## **How to run locally:

### 1. Clone the Repository
```bash
git clone https://github.com/kundanpdl/stock_prediction.git
```
### 2. Navigate to Project Directory
```bash
cd stock_prediction
```

### 3. Create Virtual Environment
``` bash
python -m venv venv
```
or
```bash
python3 -m venv venv
```

### 4. Activate The Virtual Environment
```bash
source venv/bin/activate
```

### 5. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 5. Run
```bash
streamlit run main.py
```

The application should start in your default browser.
---
