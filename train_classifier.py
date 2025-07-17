import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from src.utils.embedder import BERTEmbeddings, JinaEmbeddingsV3TextMatching
from src.utils.regression_head import RegressionHead
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
import numpy as np
import argparse
from sklearn.metrics import f1_score
from scipy.stats import spearmanr
import multiprocessing
import glob
import json
# multiprocessing.set_start_method('spawn', force=True)

class BERTClassifier(LightningModule):
    def __init__(self, dataset=None, input_column="input_text", target_column="score", base_model_name="UBC-NLP/ARBERT", train_batch_size=256, eval_batch_size=128):
        super().__init__()
        if 'jina' in base_model_name.lower():
            self.embedder = JinaEmbeddingsV3TextMatching(device='cuda', model_id=base_model_name) 
        else:
            self.embedder = BERTEmbeddings(device='cuda', model_id=base_model_name) 
        output_dim = self.embedder.model.config.hidden_size
        self.regression_head = RegressionHead(input_dim=output_dim, hidden_dim=1024)        
        self.dataset = dataset
        self.input_column = input_column
        self.target_column = target_column
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad(), torch.cuda.device('cuda'):      
            output = self.embedder.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # Extract and normalize the embeddings.
            embeddings = output.last_hidden_state[:, 0]
            embeddings = F.normalize(embeddings, p=2, dim=1)
        logits = self.regression_head(embeddings)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        targets = batch[self.target_column].float()
        logits = self(input_ids, attention_mask, token_type_ids)
        loss = F.mse_loss(logits.squeeze(), targets)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        targets = batch[self.target_column].float()
        logits = self(input_ids, attention_mask, token_type_ids)
        loss = F.mse_loss(logits.squeeze(), targets)
        
        # Convert predictions to class labels for F1 score
        predictions = torch.round(torch.clamp(logits.squeeze(), 0, 5)).long()
        targets_int = targets.long()
        
        # Calculate F1 score
        pred_np = predictions.cpu().numpy()
        target_np = targets_int.cpu().numpy()
        f1 = f1_score(target_np, pred_np, average='macro', zero_division=0)
        
        # Calculate Spearman correlation using continuous predictions
        # Convert to float32 before numpy conversion to handle bfloat16
        logits_np = logits.squeeze().float().cpu().numpy()
        targets_np = targets.cpu().numpy()
        spearman_corr, _ = spearmanr(logits_np, targets_np)
        
        # Handle NaN case (when all predictions or targets are identical)
        if np.isnan(spearman_corr):
            spearman_corr = 0.0
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_spearman', spearman_corr, prog_bar=True)
        
        return {'val_loss': loss, 'val_f1': f1, 'val_spearman': spearman_corr}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.regression_head.parameters(), lr=1e-3)
        return optimizer

    def tokenize_dataset(self, examples):
        output = {}
        
        output = self.embedder.tokenize(examples, self.input_column)
        output[self.target_column] = torch.tensor([example[self.target_column] for example in examples])
        return output

    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'] if 'train' in self.dataset else self.dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.tokenize_dataset,
            num_workers=0
        )

    def val_dataloader(self):
        # Use validation split if available, otherwise use a portion of train data
        if 'validation' in self.dataset:
            val_dataset = self.dataset['validation']
        elif 'val' in self.dataset:
            val_dataset = self.dataset['val']
        elif 'test' in self.dataset:
            val_dataset = self.dataset['test']
        else:
            # If no validation set, use 20% of training data
            train_dataset = self.dataset['train'] if 'train' in self.dataset else self.dataset
            val_size = int(0.2 * len(train_dataset))
            val_dataset = train_dataset.select(range(val_size))
        
        return DataLoader(
            val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.tokenize_dataset,
            num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.tokenize_dataset,
            num_workers=0
        )
    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {k.replace('regression_head.', ''): v for k, v in checkpoint['state_dict'].items() if 'regression_head' in k}

        checkpoint['hparams'] = {'input_dim': self.embedder.model.config.hidden_size, 'hidden_dim': 1024}
        return checkpoint

def get_dataset(distilled_model_name):
    examples = []
    files = glob.glob(f"/ibex/ai/home/alyafez/.cache/jql/**/results.jsonl")
    for file in files:
        with open(file, "r") as f:
            # get first line only
            for i,line in enumerate(f):
                data = json.loads(line)
                if i == 0:
                    if data['model'] != distilled_model_name:
                        print(f"Skip: {data['model']}")
                        break
                    else:
                        print(f"Found: {data['model']}")
                examples.append({
                    "input_text": data['input_text'],
                    "score": int(data['score']),
                })
    if len(examples) == 0:
        raise ValueError(f"No dataset found for {distilled_model_name}")

    dataset = Dataset.from_list(examples)
    # split the dataset into train and test
    train_size = int(0.9 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, len(dataset)))
    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    return dataset

def main(args):
    dataset = get_dataset(args.distilled_model_name)
    dataset = dataset.map(lambda x: {args.target_column: np.clip(int(x[args.target_column]), 0, 5)}, num_proc=4)
    dataset = dataset.filter(lambda x: x[args.target_column] != 5)
    
    print(dataset)
    model = BERTClassifier(
        dataset=dataset, 
        input_column=args.input_column, 
        target_column=args.target_column,
        base_model_name=args.base_model_name,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size
    )

    # gpu type detection
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        if 'a100' in gpu_name:
            precision = "bf16"
        else:
            precision = "32"
    else:
        precision = "32"

    print(f"Using precision: {precision}")
    num_steps = len(dataset['train']) // args.train_batch_size
    trainer = Trainer(
        max_epochs=args.epochs,
        val_check_interval=num_steps // 4,
        precision=precision,
        num_sanity_val_steps=0    
    )
    trainer.fit(model)
    trainer.save_checkpoint(f"{args.checkpoint_dir}/regression_head.ckpt")    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name", type=str, default="UBC-NLP/ARBERT"
    )
    parser.add_argument(
        "--distilled_model_name",
        type=str,
        default=f"gemma-3-27b-it",
    )
    parser.add_argument("--input_column", type=str, default="input_text")
    parser.add_argument("--target_column", type=str, default="score")
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()
    args.checkpoint_dir = args.base_model_name.replace('/', '__')+'@'+args.distilled_model_name.replace('/', '__')
    print(args)

    main(args)

