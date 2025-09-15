# 🏡 Propmind Backend – Housing Price Prediction API

This is the backend service for **Propmind**, a property price prediction tool for the UK housing market.  
It provides machine learning–based valuations, explanations, and market insights via a REST API.

---

## 🚀 Features

- **Price Prediction** – estimate property prices with confidence intervals (`/predict`)
- **Explainability** – SHAP-style breakdown of key features influencing predictions (`/explain`)
- **Data Interpretation** – natural language queries over data (`/data_nlp`)
- **Heatmaps** – aggregated sales + predictions per cluster/postcode (`/heatmap`)
- **Similar Properties** – find comparable properties (`/similar`)
- **Reports** – generate PDF market/property reports (`/report`)

---

## 🛠️ Tech Stack

- [Python 3.10+](https://www.python.org/)  
- [FastAPI](https://fastapi.tiangolo.com/) – API framework  
- [XGBoost](https://xgboost.readthedocs.io/) / [LightGBM](https://lightgbm.readthedocs.io/) – ML models  
- [Polars](https://www.pola.rs/) – fast data processing  
- [SHAP](https://shap.readthedocs.io/) – explainability  
- [ReportLab](https://www.reportlab.com/) – PDF generation  
- [AWS] (planned) – deployment and databases  

---

## ⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/rest_api.git
cd rest_api


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

▶️ Running the API

Start the FastAPI server:

uvicorn main:app --reload


The API will be available at:

http://127.0.0.1:8000


Swagger UI docs are automatically generated at:

http://127.0.0.1:8000/docs

📡 Example Usage
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"address": "10 Downing Street, London", "bedrooms": 3}'


Response:

{
  "predicted_price": 725000,
  "confidence_interval": [690000, 760000]
}

