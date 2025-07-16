from datasets import load_dataset, Dataset

def truncate(self, examples):
    encoded = self.tokenizer(examples["text"], padding=False, truncation=True, max_length=self.max_length)
    examples["num_tokens"] = [len(ids) for ids in encoded["input_ids"]]
    decoded = self.tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=True)
    examples["text"] = decoded
    return examples

fw = load_dataset("/ibex/ai/project/c2254/arabic_raw_data/fineweb2_arabic/data/arb_Arab", split = 'train', streaming=True)

print(fw[0])
fw = fw.map(truncate, batched=True, batch_size=1000)

