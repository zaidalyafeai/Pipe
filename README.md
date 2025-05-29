# JQL-Annotation-Pipeline

## Installation 

Clone repository 
```
git clone https://github.com/JQL-AI/JQL-Annotation-Pipeline.git
cd ./JQL-Annotation-Pipeline/
```

Install torch requirements for your specific CUDA driver (here CUDA 12.6). 

`pip install -r torch_requirements.txt --index-url https://download.pytorch.org/whl/cu126`

Install other dependencies: 

`pip install -r requirements.txt`

## Usage Example
A minimal usage example using our [datatrove](https://github.com/huggingface/datatrove) pipeline could look like the snippet below. This code will automatically load our pre-trained Edu annotators from [huggingface](https://huggingface.co/Jackal-AI/JQL-Edu-Heads).

When providing the `JQLAnnotator` with a `stats-writer` the pipeline will write only the meta-data of each document to a seperate directory in addition to the regular data writer. 
We found this to be particularly useful for annotation analysis without needing to load the actual text documents. 

```python 
from datatrove_jql_annotator import JQLAnnotator, stats_adapter
from datatrove.pipeline.readers import ParquetReader
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.writers import ParquetWriter, JsonlWriter


pipeline = [
            ParquetReader(
                data_folder='webdata_inputs', # Change to your input directory
                glob_pattern='*.parquet',
            ),          
            JQLAnnotator(
                stats_writer=JsonlWriter(
                    output_folder=f'jql_outputs/stats', # Change to your output directory
                    adapter=stats_adapter,
                    expand_metadata=True,
                    ),
            ),
            ParquetWriter(
                output_folder=f'jql_outputs/data' # Change to your output directory
            ),
    ]
stage = LocalPipelineExecutor(
    pipeline,
    tasks=2, # n_tasks to be executed across all machines
    local_tasks=2, # n_tasks to be executed on this machine (e.g. the number of available gpus)
    local_rank_offset=0, # determines the first rank/task of this machine (has to be adapted per machine)
    logging_dir=f'./logs/test/jql'
)
```

Alternatively, you can use the JQL Annotators by modifing the code below 

```python 
from utils.regression_head import RegressionHead
from transformers.utils.hub import cached_file
from utils.embedder import get_embedder_instance
import torch

# load embedder
device = 'cuda'
embedder = get_embedder_instance('Snowflake/snowflake-arctic-embed-m-v2.0', device, torch.bfloat16)
# load JQL Edu annotation heads
regression_head_checkpoints = {
                'Edu-JQL-Gemma-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-gemma-snowflake-balanced.ckpt'),
                'Edu-JQL-Mistral-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-mistral-snowflake-balanced.ckpt'),
                'Edu-JQL-Llama-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-llama-snowflake-balanced.ckpt'),
            }
regression_heads = {}
for name, path in regression_head_checkpoints.items():
    regression_heads[name] = RegressionHead.load_from_checkpoint(path, map_location=device).to(torch.bfloat16)

# Given a single document
doc = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua'
embeddings = embedder.embed([doc])
scores = {}
with torch.no_grad():
    for name, regression_head in regression_heads.items():
        scores[f'score_{name}'] = regression_head(embeddings).cpu().squeeze(1)
```
## 📖 Citation
Please cite as
```bibtex
@article{ali2025judging,
    title     = {Judging Quality Across Languages: A Multilingual Approach to Pretraining Data Filtering with Language Models},
    author    = {
      Mehdi Ali,
      Manuel Brack,
      Max Lübbering,
      Elias Wendt,
      Abbas Goher Khan,
      Richard Rutmann,
      Alex Jude,
      Maurice Kraus,
      Alexander Arno Weber,
      Felix Stollenwerk,
      David Kaczér,
      Florian Mai,
      Lucie Flek,
      Rafet Sifa,
      Nicolas Flores-Herr,
      Joachim Köhler,
      Patrick Schramowski,
      Michael Fromm,
      Kristian Kersting
    },
    year      = {2025},
    journal   = {arXiv preprint arXiv:2505:22232}
  }
```
