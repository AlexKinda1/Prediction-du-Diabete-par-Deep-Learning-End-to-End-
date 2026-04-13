from fastapi import FastAPI
from deployment.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title='projet_deep_learning API')

@app.get('/')
def health_check():
    return {'status': 'ok'}

@app.post('/predict', response_model=PredictionResponse)
def predict(data: PredictionRequest):
    # TODO : charger modèle avec model_loader et faire inférence
    return PredictionResponse(predictions=[0])
