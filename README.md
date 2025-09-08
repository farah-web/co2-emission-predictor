🌍 CO₂ Emission Predictor

A machine learning regression project that predicts vehicle CO₂ emissions based on engine specifications, transmission type, fuel type, and fuel consumption metrics. This project includes exploratory data analysis (EDA), model training, and deployment on Streamlit Cloud.

🔗 Live App: CO₂ Emission Predictor

📌 Project Overview

Transportation is one of the major contributors to greenhouse gas emissions worldwide. This project aims to:

Analyze vehicle CO₂ emissions dataset

Explore relationships between engine size, cylinders, fuel type, and emissions

Train machine learning models to predict emissions

Deploy an interactive Streamlit web app for real-time predictions

📂 Project Structure
├── CO2_Emmission_ML_Project.ipynb   # Jupyter Notebook (EDA + Model training)
├── co2_emissions.csv                # Dataset (if included)
├── app.py / co2app.py               # Streamlit app script
├── requirements.txt                 # Dependencies
└── README.md                        # Project documentation

⚙️ Features

✅ Data preprocessing & handling categorical variables
✅ Rare-category grouping (threshold-based) for make, model, and vehicle_class
✅ Multiple regression models tested (Linear Regression, Random Forest, etc.)
✅ Interactive Streamlit app with dropdowns & sliders for input
✅ Visualizations for EDA and feature insights

📊 Dataset

The dataset contains details about different vehicles:

Numerical features: Engine Size, Cylinders, Fuel Consumption (City, Hwy, Comb)

Categorical features: Make, Model, Vehicle Class, Transmission, Fuel Type

Target variable: CO₂ Emissions (g/km)

🚀 Deployment

The project is deployed using Streamlit Cloud.

🔗 Click here to try the live app

📦 Installation

Clone this repo and install dependencies:

git clone https://github.com/your-username/co2-emission-predictor.git
cd co2-emission-predictor
pip install -r requirements.txt


Run locally:

streamlit run co2app.py

🖥️ Usage

Open the app

Select vehicle parameters (Make, Model, Transmission, Fuel Type, etc.)

Adjust engine size, cylinders, and fuel consumption sliders

Get predicted CO₂ emissions instantly
