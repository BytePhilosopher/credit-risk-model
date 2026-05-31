"""FastAPI application for serving credit-risk predictions.

Implemented in Task 6: loads the model from the MLflow registry and exposes a
/predict endpoint validated by the Pydantic schemas in pydantic_models.py.
"""

from fastapi import FastAPI

app = FastAPI(title="Credit Risk Model API")


@app.get("/")
def health_check():
    return {"status": "ok"}
