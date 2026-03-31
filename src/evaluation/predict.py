import torch

def batch_predict(model, dataloader, device='cpu'):
    
    model.eval()
    
    preds = []
    
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            logits = model(x)
            preds += logits.argmax(dim=1).tolist()
    return preds
