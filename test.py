from src.utils.regression_head import RegressionHead
from transformers.utils.hub import cached_file
from src.utils.embedder import get_embedder_instance
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from scipy.stats import spearmanr
import glob
from tabulate import tabulate
def predict(examples, embedder, regression_head, name):
    embeddings = embedder.embed(examples['text'])
    with torch.no_grad():
        scores = regression_head(embeddings).cpu().squeeze(1).float().numpy()
    examples[f'{name}_score'] = scores
    return examples

data = load_dataset('JQL-AI/JQL-Human-Edu-Annotations', split='test')
data = data.select(range(511))
files = glob.glob('**/regression_head.ckpt')
# load embedder
device = 'cuda'

# load JQL Edu annotation heads
regression_head_checkpoints = {
                'Edu-JQL-Gemma-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-gemma-snowflake-balanced.ckpt'),
                'Edu-JQL-Mistral-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-mistral-snowflake-balanced.ckpt'),
                'Edu-JQL-Llama-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-llama-snowflake-balanced.ckpt'),
            }

for file in files:
    name = file.split('/')[0].replace('__', '/')
    regression_head_checkpoints[name] = file

regression_heads = {}
embedder = {}
for name, path in regression_head_checkpoints.items():
    if 'JQL' in name:
        embedder[name] = get_embedder_instance('Snowflake/snowflake-arctic-embed-m-v2.0', device, torch.bfloat16)
    else:
        embedder[name] = get_embedder_instance(name, device)
        embedder[name].model.to(torch.bfloat16)

    regression_heads[name] = RegressionHead.load_from_checkpoint(path, map_location=device).to(torch.bfloat16)

# Given a single document
for name, regression_head in regression_heads.items():
    data = data.map(predict, fn_kwargs={'embedder': embedder[name], 'regression_head': regression_head, 'name': name}, batched=True, batch_size=100)



# calculate scores
results = []
for name, _ in regression_heads.items():
    x = [round(float(score)) for score in data[f'{name}_score']]
    y = [round(float(score)) for score in data['score']]
    f1 = f1_score(x, y, average='macro')
    spearman_corr = spearmanr(x, y).correlation
    results.append({'name': name, 'f1': f1, 'spearman_corr': spearman_corr})

# sort by spearman_corr
results.sort(key=lambda x: x['spearman_corr'], reverse=True)

# show table and round to 2 decimal places
print(tabulate(results, headers='keys', tablefmt='grid', floatfmt='.2f'))

