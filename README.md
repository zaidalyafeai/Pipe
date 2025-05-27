# JQL-Annotation-Pipeline


## Usage Example

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