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
A minimal usage example could look like the snippet below. This code will automatically load our pre-trained Edu annotators from [huggingface](https://huggingface.co/Jackal-AI/JQL-Edu-Heads).

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