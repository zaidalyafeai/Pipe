from src.utils.regression_head import RegressionHead
from transformers.utils.hub import cached_file
from src.utils.embedder import get_embedder_instance
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from scipy.stats import spearmanr

data = load_dataset('JQL-AI/JQL-Human-Edu-Annotations', split='test')
data = data.select(range(511))

# load embedder
device = 'cuda'

# load JQL Edu annotation heads
regression_head_checkpoints = {
                'Edu-JQL-Gemma-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-gemma-snowflake-balanced.ckpt'),
                'Edu-JQL-Mistral-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-mistral-snowflake-balanced.ckpt'),
                'Edu-JQL-Llama-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-llama-snowflake-balanced.ckpt'),
                'Edu-ARBERT': 'arbert_classifier/regression_head.ckpt',
            }
regression_heads = {}
embedder = {}
for name, path in regression_head_checkpoints.items():
    if 'JQL' in name:
        embedder[name] = get_embedder_instance('Snowflake/snowflake-arctic-embed-m-v2.0', device, torch.bfloat16)
    else:
        embedder[name] = get_embedder_instance('UBC-NLP/ARBERT', device, torch.bfloat16)

    regression_heads[name] = RegressionHead.load_from_checkpoint(path, map_location=device).to(torch.bfloat16)

scores = {}
# Given a single document
for name, regression_head in regression_heads.items():
    if name not in scores:
        scores[f'{name}_score'] = []
    for doc in data['text']:
        embeddings = embedder[name].embed([doc])
        with torch.no_grad():
            scores[f'{name}_score'].append(regression_head(embeddings).cpu().squeeze(1).float().numpy()[0])



# calculate scores  
for name, _ in regression_heads.items():
    x = [round(float(score)) for score in data['score']]
    y = [round(float(score)) for score in scores[f'{name}_score']]
    print(x)
    print(y)
    f1 = f1_score(x, y, average='macro')
    spearman_corr = spearmanr(x, y).correlation
    print(f'{name} F1 score: {f1}, Spearman correlation: {spearman_corr}')

