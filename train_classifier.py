import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from src.utils.embedder import BERTEmbeddings, JinaEmbeddingsV3TextMatching
from src.utils.regression_head import RegressionHead
from datasets import load_from_disk, load_dataset
import numpy as np
import argparse
from sklearn.metrics import f1_score
from scipy.stats import spearmanr
import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)

class BERTClassifier(LightningModule):
    def __init__(self, dataset=None, input_column="input_text", target_column="score", base_model_name="UBC-NLP/ARBERT"):
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
        tokenized = self.embedder.model.encode(
                [example[self.input_column] for example in examples],
                padding=True,
                truncation=True,
                max_length=512,
                # return_tensors='pt',
                task='text-matching')
        output['input_ids'] = torch.tensor(tokenized, dtype=torch.long)
        output['attention_mask'] = torch.ones_like(output['input_ids'], dtype=torch.long)
        output['token_type_ids'] = torch.zeros_like(output['input_ids'], dtype=torch.long)
        output[self.target_column] = torch.tensor([example[self.target_column] for example in examples])
        return output

    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'] if 'train' in self.dataset else self.dataset,
            batch_size=256,
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
        else:
            # If no validation set, use 20% of training data
            train_dataset = self.dataset['train'] if 'train' in self.dataset else self.dataset
            val_size = int(0.2 * len(train_dataset))
            val_dataset = train_dataset.select(range(val_size))
        
        return DataLoader(
            val_dataset,
            batch_size=128,
            shuffle=False,
            collate_fn=self.tokenize_dataset,
            num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=128,
            shuffle=False,
            collate_fn=self.tokenize_dataset,
            num_workers=0
        )
    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {k.replace('regression_head.', ''): v for k, v in checkpoint['state_dict'].items() if 'regression_head' in k}

        checkpoint['hparams'] = {'input_dim': self.embedder.model.config.hidden_size, 'hidden_dim': 1024}
        return checkpoint

def main(args):
    dataset = load_from_disk(args.dataset_name)
    dataset = dataset.map(lambda x: {args.target_column: np.clip(int(x[args.target_column]), 0, 5)}, num_proc=4)
    dataset = dataset.filter(lambda x: x[args.target_column] != 5)
    
    model = BERTClassifier(
        dataset=dataset, 
        input_column=args.input_column, 
        target_column=args.target_column,
        base_model_name=args.base_model_name
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
    trainer = Trainer(
        max_epochs=1,
        val_check_interval=500,
        precision=precision,
        num_sanity_val_steps=0    
    )
    trainer.fit(model)
    trainer.save_checkpoint(f"{args.checkpoint_dir}/regression_head.ckpt")    
    
if __name__ == "__main__":
    base_path = '/ibex/ai/home/alyafez/curator/examples/custom'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name", type=str, default="UBC-NLP/ARBERT"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=f"{base_path}/annotated_dataset",
    )
    
    parser.add_argument("--input_column", type=str, default="input_text")
    parser.add_argument("--target_column", type=str, default="score")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()
    args.checkpoint_dir = args.base_model_name.replace('/', '__')
    print(args)

    main(args)

