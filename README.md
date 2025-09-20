📌 Project Overview
Insurance Premium Prediction System

This project predicts the insurance premium category of an individual based on demographic, health, and lifestyle features. It is an end-to-end ML system consisting of:

- Feature Engineering & Model Training (Random Forest Classifier)

- Model Serialization with Pickle

- API for Model Serving using FastAPI

- Frontend for User Interaction using Streamlit

The system allows users to input their personal and health details through a web interface and receive an estimated premium category prediction.


Project Workflow
1. Data Preparation

- Loaded dataset insurance.csv using Pandas.
- Explored unique values and class distribution.

2. Feature Engineering

Created new features from existing ones to improve model performance:

- BMI: BMI = weight/(height)**2
- Age Group: Categorized into young, adult, middle_aged, and senior.
- Lifestyle Risk: Based on smoking status and BMI (low, medium, high).
- City Tier: Classified cities into tier_1, tier_2, or tier_3.

Final feature set:

- income_lpa
- occupation
- bmi
- age_group
- lifestyle_risk
- city_tier
Target: insurance_premium_category

3. Preprocessing & Model Training

- Categorical Features -> OneHotEncoder
- Numerical Features -> Passed directly
- Model: RandomForestClassifier (random_state=42)
- Pipeline: Preprocessing + Classifier


4. Model Serialization

Saved trained pipeline as a .pkl file using pickle.

5. FastAPI Backend

- Exposes REST API for prediction.
- Loads model.pkl and performs inference.

Endpoints:
- POST /predict: Accepts JSON input and returns predicted premium category.

6. Streamlit Frontend

- Provides a user-friendly UI.
- Users input their details (age, weight, height, city, smoker status, etc.).
- Data is processed → sent to FastAPI backend → prediction displayed on UI.