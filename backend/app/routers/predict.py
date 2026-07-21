from fastapi import APIRouter

from app.ml.predict_core import predict
from app.schemas import PatientInput, PredictResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict_endpoint(patient: PatientInput) -> dict:
    """
    Hand-off 1, made real: this just calls Track A's predict() function
    with the request body turned into a plain dict, and returns whatever
    it returns -- no extra logic here on purpose. All the real work
    (validation, model inference, SHAP) lives in predict_core.py.
    """
    return predict(patient.model_dump())
