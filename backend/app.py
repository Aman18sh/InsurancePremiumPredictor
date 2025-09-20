from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pickle
import pandas as pd


tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]

# importing the ml model

with open('model.pkl','rb') as f:
    model = pickle.load(f)

app = FastAPI()

# building a pydantic model to validata incoming data

class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=0,lt=120,description='Age of the user')]
    weight: Annotated[float,Field(..., gt=0,description='weight of the user')]
    height: Annotated[float,Field(...,gt=0,lt=2.5,description='Height of the user')]
    income_lpa: Annotated[float,Field(...,gt=0,description='Annual salary of the user in lpa')]
    smoker: Annotated[bool,Field(...,description='Is user a smoker')]
    city: Annotated[str, Field(...,description='The city that the user belongs to')]
    occupation: Annotated[Literal['retired', 'freelancer', 'student', 'government_job',
       'business_owner', 'unemployed', 'private_job'],Field(...,description='Occupation of the user')]


    @computed_field
    @property
    def bmi(self) -> float:
        bmi = self.weight/(self.height**2)
        return bmi

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi>30:
            return 'high'
        elif self.smoker or self.bmi > 27:
            return 'medium'
        else:
            return 'low'
        
    
    @computed_field
    @property

    def age_group(self) -> str:
        if self.age < 25:
            return 'young'
        elif self.age < 45:
            return 'adult'
        elif self.age < 60:
            return 'middle_aged'
        else:
            return 'senior'
        
    @computed_field
    @property
    def city_tier(self) ->str:
        if self.city in tier_1_cities:
            return 'tier_1'
        elif self.city in tier_2_cities:
            return 'tier_2'
        else:
            return 'tier_3'

# Making the end point for prediction
@app.post('/predict')

def  predict_premium(data: UserInput):
    # input should be in panda dataframe as our model trained on pandas dataframe

    input_df = pd.DataFrame([{
        'bmi':data.bmi,
        'age_group':data.age_group,
        'lifestyle_risk':data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa':data.income_lpa,
        'occupation':data.occupation
                }])
    
    prediction = model.predict(input_df)[0]

    return JSONResponse(status_code=200,content={'Predicted_Category':prediction})

    
         

    
