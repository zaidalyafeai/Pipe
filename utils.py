from src.utils.embedder import get_embedder_instance
from src.utils.regression_head import RegressionHead
import torch

def compute_scores(examples, embedder, regression_head, name):
    embeddings = embedder.embed(examples['text'])
    with torch.no_grad():
        scores = regression_head(embeddings).cpu().squeeze(1).float().numpy()
    examples['score'] = scores
    return examples

def load_classifier(model_path, device):
    name = model_path.split('/')[0].replace('__', '/')
    embedder = get_embedder_instance(name, device)
    embedder.model.to(torch.bfloat16)

    regression_head = RegressionHead.load_from_checkpoint(model_path, map_location=device).to(torch.bfloat16)
    return embedder, regression_head
