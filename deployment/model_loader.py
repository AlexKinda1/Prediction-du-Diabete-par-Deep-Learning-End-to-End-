import torch


def load_model(model_uri: str):
    # TODO : téléchargement depuis MLflow/W&B
    model = torch.load(model_uri, map_location='cpu')
    model.eval()
    return model
