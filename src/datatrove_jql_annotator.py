from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from utils.regression_head import RegressionHead
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.data import DocumentsPipeline, Document
from datatrove.utils.batching import batched
from transformers.utils.hub import cached_file
import dataclasses
import contextlib

from torch import no_grad, cuda, bfloat16
import itertools
from datatrove.utils.logging import logger

from utils.embedder import get_embedder_instance
from abc import ABC
       
        
def stats_adapter(writer: DiskWriter, document: Document, expand_metadata=True) -> dict:
    """
    The datatrove adapter to write stats metadata without the actual document text

    Args:
        writer: the diskwriter
        document: a datatrove document

    Returns: a dictionary of metadata without the text field

    """
    data = {key: val for key, val in dataclasses.asdict(document).items() if val and key != "text"}
    if writer.expand_metadata and "metadata" in data:
            data |= data.pop("metadata")
    return data


class JQLAnnotator(PipelineStep, ABC):
    """
    A pipeline step for annotating text data for using a combination
    of a pre-trained multilingual embedding model and a custom regression heads.

    The process involves:
    1. Embedding text documents using a specified multilingual embedding model.
    2. Passing these embeddings through multiple fine-tuned regression head to predict a score. (E.g. for educational value)

    The score reflects the likelihood of a text being "educational" or high-quality
    for LLM pretraining, as distilled from larger language models' judgments.

    """
    
    name = "📊 Score"
    type = "🔢 - JQL-ANNOTATOR"

    def __init__(
        self,
        embedder_model_id: str = 'Snowflake/snowflake-arctic-embed-m-v2.0',
        regression_head_checkpoints: Optional[dict[str, str]] = None,
        batch_size: int = 1_000,
        device_overwrite: Optional[str] = None,
        stats_writer: DiskWriter = None,
    ):
        """
        Initializes the JQLAnnotator.

        Args:
            embedder_model_id (str): The identifier for the multilingual embedding model
                                     to be used ('Alibaba-NLP/gte-multilingual-base',
                                     'Snowflake/snowflake-arctic-embed-m-v2.0', or 'jinaai/jina-embeddings-v3').
                                     This model is responsible for converting text into
                                     numerical representations.
            regression_head_checkpoints (dict[str, str], optional): A dictionary where keys are
                                                          identifiers and values are file paths to the
                                                          PyTorch Lightning checkpoints
                                                          (.ckpt files) of the RegressionHead
                                                          model. Metadata names will be written based
                                                          on provided head names. If set to `None` will
                                                          load default regression heads from the JQL Paper.
            batch_size (int): The number of text samples to process in a single
                                        batch during embedding and annotation. Larger batch
                                        sizes can improve throughput but require more memory.
                                        Defaults to 1,000.
            device_overwrite (str, optional): An optional string specifying the device
                                              (e.g., 'cuda', 'cuda:0', 'cpu') on which
                                              the models should be loaded and computations
                                              performed. If None, processes will be devided over
                                              all available GPUs.
            stats_writer (DiskWriter, optional): An instance of a DiskWriter (from Datatrove)
                                                or a compatible class, used to log and save
                                                document metadata seperately from the document text.
        """
        super().__init__()
        
        self.embedder_model_id = embedder_model_id
        if regression_head_checkpoints is None: 
            if self.embedder_model_id != 'Snowflake/snowflake-arctic-embed-m-v2.0':
                raise ValueError(f'No custom regression heads were specified, but default JQL Edu heads are not compatible with embedding model {self.embedder_model_id}.')
            logger.info('No custom regression heads specified. Using default JQL Edu heads.')
            self.regression_head_checkpoints = {
                'Edu-JQL-Gemma-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-gemma-snowflake-balanced.ckpt'),
                'Edu-JQL-Mistral-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-mistral-snowflake-balanced.ckpt'),
                'Edu-JQL-Llama-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-llama-snowflake-balanced.ckpt'),
            }
        else:
            self.regression_head_checkpoints = regression_head_checkpoints
        self.batch_size = batch_size        
        self.device_overwrite = device_overwrite
        self.stats_writer = stats_writer

    def run(self, doc_pipeline: DocumentsPipeline, rank: int = 0, world_size: int = 1, **kwargs) -> DocumentsPipeline:
        """
        Args:
          data: DocumentsPipeline:
          rank: int: id of this task (Default value = 0)
          world_size: int: total number of parallel tasks (Default value = 1)

        Returns:
          DocumentsPipeline: The pipeline with updated documents, each having a new or updated `edu_score` in its metadata.

        """
                
        if not cuda.is_available():
            logger.warning('CUDA is not available, using CPU')
            device = 'cpu'
        else:
            if self.device_overwrite is None:
                device_count = cuda.device_count()
                cuda_device_id = rank % device_count # distribute tasks over available gpus
                device = f'cuda:{cuda_device_id}'
            else:
                logger.info(f'Overwriting device for all ranks with {self.device_overwrite}')
                device=f'cuda:{self.device_overwrite}'


        # instantiate EMBEDDER
        embedder = get_embedder_instance(self.embedder_model_id, device, bfloat16)
        
            
        self.regression_heads = {}
        for name, path in self.regression_head_checkpoints.items():
            self.regression_heads[name] = RegressionHead.load_from_checkpoint(path, map_location=device).to(bfloat16)
        # instantiate REGRESSION HEAD
        #regression_head = RegressionHead.load_from_checkpoint(self.regression_head_checkpoint_path, map_location=device).to(bfloat16)
        
        # profiling_dict = defaultdict(list)
        # pipeline loop
        with self.stats_writer if self.stats_writer else contextlib.nullcontext() as writer:

            for doc_batch in batched(doc_pipeline, self.batch_size):
                
                with self.track_time(unit='batch'):
                    embeddings = embedder.embed([doc.text for doc in doc_batch])


                    scores = {}
                    with no_grad():
                        for name, regression_head in self.regression_heads.items():
                            scores[f'score_{name}'] = regression_head(embeddings).cpu().squeeze(1)

                    for batch_idx, doc in enumerate(doc_batch):
                        for name, score in scores.items():
                            doc.metadata[name] = score[batch_idx].item()
                        
                        if writer:
                            writer.write(doc, rank)
                        yield doc

