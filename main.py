from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from typing import Optional, Annotated
import sys


# Fix for loading the saved scikit-learn model
from sklearn._loss import _loss as sklearn_loss

sys.modules.setdefault("_loss", sklearn_loss)

# Create FastAPI application
app = FastAPI(title="Loan Default Risk Predictor")

# Load trained ML model
model = joblib.load("gradient_boosting_model.joblib")

# Model configuration
FEATURES = [
    "credit_utilization",
    "age",
    "late_30_59",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "late_90_plus",
    "real_estate_loans",
    "late_60_89",
    "dependents"
]

CUTOFF = 0.20

CAPS = {
    "credit_utilization": 1.0929557681400022,
    "debt_ratio": 4979.040000000037,
    "monthly_income": 23000.0
}

INCOME_MEDIAN = 5400.0

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Input data model
class ApplicantData(BaseModel):

    credit_utilization: float = Field(...,ge=0,description="Credit utilization ratio")
    age: int = Field(...,ge=18,le=110,description="Age of the applicant")
    late_30_59: int = Field(...,ge=0,description="Times 30-59 days late")
    late_60_89: int = Field(...,ge=0,description="Times 60-89 days late")
    late_90_plus: int = Field(...,ge=0,description="Times 90+ days late")
    debt_ratio: float = Field(...,ge=0,description="Debt ratio")
    monthly_income: Optional[float] = Field(None,ge=0,description="Monthly income")
    open_credit_lines: int = Field(...,ge=0,description="Number of open credit lines")
    real_estate_loans: int = Field(...,ge=0,description="Number of real estate loans")
    dependents: Optional[int] = Field(None,ge=0,description="Number of dependents")

# Home page
@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )

# Prediction endpoint
@app.post("/predict")
def predict(
    request: Request,
    data: Annotated[ApplicantData, Form()]
):

    # Convert submitted form data to dictionary
    row = data.model_dump()

    # Handle missing optional values
    if row["dependents"] is None:
        row["dependents"] = 0

    if row["monthly_income"] is None:
        row["monthly_income"] = INCOME_MEDIAN

    # Cap extreme values
    for col, cap in CAPS.items():
        row[col] = min(float(row[col]), cap)

    # Create DataFrame in correct feature order
    input_data = pd.DataFrame([row])[FEATURES]

    # Get prediction probability
    probability = float(model.predict_proba(input_data)[0, 1])
    percentage = round(probability * 100,2)
    cutoff_percentage = int(CUTOFF * 100)

    # Determine risk
    if probability >= CUTOFF:

        result = (
            f"High Risk (Flagged)! "
            f"Default Probability: {percentage}% "
            f"(Cutoff: {cutoff_percentage}%)"
        )

        status = "flagged"

    else:

        result = (
            f"Low Risk (Passed). "
            f"Default Probability: {percentage}% "
            f"(Cutoff: {cutoff_percentage}%)"
        )

        status = "safe"

    # Display result on HTML page
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "result": result,
            "status": status
        }
    )
