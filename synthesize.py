import os
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
import time
from requests.exceptions import ConnectionError, RequestException
from urllib3.exceptions import NewConnectionError, MaxRetryError
from openai import OpenAI, AsyncOpenAI
import argparse
import json
from transformers import AutoTokenizer
import asyncio
import hashlib
from tqdm import tqdm
from glob import glob


load_dotenv("../../.env")

SYSTEM_PROMPT = """
You are given an extract from a webpage in Arabic:
Write an informative and insightful blog post in Arabic that expands upon the extract above, within the context of the same topic.
Your post should delve into the nuances of the topic, offering fresh perspectives and deeper analysis.
Aim to:
Inform: Provide valuable, well-researched information that educates the reader.
Engage: Write in a conversational tone that connects with the audience, making complex ideas accessible.
Illustrate: Use examples, anecdotes, or personal experiences to bring the topic to life. Do not give a title and do not start with sentences like "Have you ever..." or "Hello dear readers..", simply write the content without these introductory phrases.
Extract: 
"""


import re
    
def check_server_status(model, HOST, PORT):
        url = f"http://{HOST}:{PORT}/v1"
        try:
            client = OpenAI(
                api_key="EMPTY",
                base_url=url,
            )
            print("running inference")
            chat_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "ping"},
                ]
            )
            print("inference done")
            print(chat_response)
            return True
        except ConnectionError as e:
            # This catches both ConnectionRefusedError and other connection issues
            if isinstance(e.args[0], MaxRetryError) and isinstance(e.args[0].reason, NewConnectionError):
                # This is the specific case of connection refused
                return False
            print(f"Connection error: {e}")
            return False
        except RequestException as e:
            print(f"Request error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False


def process_json(response):
    response = response.replace("```json", "").strip()
    response = response.replace("```", "").strip()
    return response

def extract_json(response):
        pattern = r"<output>(.*?)</output>"
        # Search for the pattern <output>...</output>
        match = re.search(pattern, response, re.DOTALL)
        default = {
            "score": 0,
            "reasoning": ""
        }
        if match:
            # Extract the content between the tags
            json_str = match.group(1).strip()
            try:
                # Parse the string to a JSON object
                json_data = json.loads(json_str)
                return json_data
            except json.JSONDecodeError:
                # Return None if JSON parsing fails
                return default
        else:
            return default

class Prompter():
    def __init__(self, model: str, max_tokens: int = 4096, HOST: str = "localhost", PORT: int = 8787, output_path: str = "/ibex/ai/home/alyafez/.cache/curator", 
    overwrite: bool = False, num_examples: int = 50_000, shard: int = 0, num_shards: int = 1, file_name: str = "000_00000.json"):
        self.model = model
        if 'gemma' in model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(f"google/{model}")
        elif 'qwen' in model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{model}")
        elif 'fanar' in model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(f"QCRI/{model}")
        elif 'allam' in model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(f"ALLaM-AI/{model}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        print(f"Using tokenizer: {self.tokenizer}")
        self.SYSTEM_PROMPT_TOKENS = len(self.tokenizer.encode(SYSTEM_PROMPT)) + 21 # 21 is the number of tokens in the system prompt
        self.MAX_OUTPUT_TOKENS =  max_tokens // 2
        self.max_length = max_tokens - self.SYSTEM_PROMPT_TOKENS - self.MAX_OUTPUT_TOKENS
        print('max_length', self.max_length)
        print(f"MAX_TOKENS: {max_tokens}")
        print(f"SYSTEM_PROMPT_TOKENS: {self.SYSTEM_PROMPT_TOKENS}")
        print(f"MAX_OUTPUT_TOKENS: {self.MAX_OUTPUT_TOKENS}")
        self.HOST = HOST
        self.PORT = PORT
        self.client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=f"http://{HOST}:{PORT}/v1",
        )
        self.output_folder = os.path.join(output_path, self.generate_hash_filename())
        os.makedirs(self.output_folder, exist_ok=True)
        self.output_file = os.path.join(self.output_folder, file_name)

        if not os.path.exists(self.output_file) or overwrite:
            with open(self.output_file, 'w') as f:
                f.write("")
        self.num_examples = num_examples
        self.shard = shard
        self.num_shards = num_shards
        self.file_name = file_name  
    def get_num_tokens(self, text):
        encoded = self.tokenizer(text, padding=False, truncation=False)
        return len(encoded["input_ids"])
    
    def truncate(self, examples):
        encoded = self.tokenizer(examples["text"], padding=False, truncation=True, max_length=self.max_length)
        examples["num_tokens"] = [len(ids) for ids in encoded["input_ids"]]
        decoded = self.tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=True)
        examples["text"] = decoded
        return examples
    
    def create_dataset(self, fw):
        examples = []
        pbar = tqdm(iter(fw), total=self.num_examples)
        for example in pbar:
            examples.append(example)
            pbar.set_description(f"Creating dataset: {len(examples)}/{self.num_examples}")
            if len(examples) >= self.num_examples:
                break
        return Dataset.from_list(examples)
    
    def wait_for_server(self):
        retry_count = 0
        MAX_RETRIES = 30  # 5 minutes with 10 second intervals
        RETRY_INTERVAL = 10
        while not check_server_status(self.model, self.HOST, self.PORT):
            if retry_count >= MAX_RETRIES:
                print("Maximum retries reached. Server failed to start.")
                exit(1)
            print(f"Waiting for the server to start... (Attempt {retry_count + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_INTERVAL)
            retry_count += 1
        print("Server started")

    def process_dataset(self, fw, batch_size=10):
        # fw = self.create_dataset(fw)
        # fw = fw.filter(
        #     lambda x, i: i % self.num_shards == self.shard, with_indices=True
        # )
        fw = fw.map(self.truncate, batched=True, batch_size=1000, num_proc=1)
        file_path = f"{self.output_folder}/{self.file_name}"

        if os.path.exists(file_path):
            # load the ids from the file
            print(f"Loading {file_path}")
            with open(file_path, 'r') as f:
                ids = [json.loads(line.strip())["id"] for line in f]
            if len(ids) > 0:
                fw = fw.filter(lambda x: x["id"] not in ids)
                print(f"Filtered dataset to {len(fw)} examples")
        self.wait_for_server()
        results = []
        pbar = tqdm(range(0, len(fw), batch_size))
        # moving average of rps and tps
        rps_list = []
        tps_list = []
        total_tokens = 0
        for i in pbar:
            batch_requests = 0
            batch_tokens = 0
            if i+ batch_size > len(fw):
                batch_size = len(fw) - i
            batch = fw.select(range(i, i+batch_size))
            start_time = time.time()
            batch_results = asyncio.run(self.process_batch(batch))
            results.extend(batch_results)
            
            for example in batch_results:
                batch_tokens += example["num_generated_tokens"]

            total_tokens += batch_tokens
            # Update RPM metrics
            batch_requests += len(batch)
            elapsed_time = time.time() - start_time
            rps = batch_requests / elapsed_time  if elapsed_time > 0 else 0
            tps = batch_tokens / elapsed_time if elapsed_time > 0 else 0
            rps_list.append(rps)
            tps_list.append(tps)
            rps = sum(rps_list) / len(rps_list)
            tps = sum(tps_list) / len(tps_list)
            # Update progress bar description
            pbar.set_description(f"Processed: {i+len(batch)}/{len(fw)} | RPS: {rps:.2f} | TPS: {tps:.2f} | TOTAL_TOKENS: {total_tokens}")
            
            with open(file_path, 'a') as f:
                for result in batch_results:
                    f.write(json.dumps(result) + '\n')

    async def process_single_example(self, example):
        """Process a single example with retries."""
        max_tries = 0
        output = {
                "text": "",
                "id": example["id"],
                "url": example["url"],
                "input_text": example["text"],
                "num_tokens": example["num_tokens"],
                "model": self.model,
            }
        while max_tries < 1:
            # Generate prompt
            prompt_text = self.prompt(example)
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,  
                messages=[
                    {"role": "system", "content": "You are an AI assistant evaluating educational content."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.1
            )
            response = response.choices[0].message.content
            try:
                result = self.parse(response)
                output = output | result
                break
            except Exception as e:
                print(e)
                print(response)
                print(example['num_tokens'])
                max_tries += 1
                time.sleep(1)
            
        return output

    def generate_hash_filename(self):
        content_to_hash = f"{SYSTEM_PROMPT}{self.model}"    
        hash_value = hashlib.sha256(content_to_hash.encode('utf-8')).hexdigest()
        return f"{hash_value[:10]}"
    
    async def process_batch(self, batch):
        tasks = [self.process_single_example(example) for example in batch]
        return await asyncio.gather(*tasks)
    
    """A recipe generator that generates recipes for different cuisines."""
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the recipe generator."""
        return SYSTEM_PROMPT + input["text"]

    def parse(self,response: str) -> dict:
        return {"text": response, "num_generated_tokens": self.get_num_tokens(response)}
            

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--mode", type=str, default="vllm")
    args.add_argument("--model", type=str, default="gemma-3-4b-it")
    args.add_argument("--num-examples", type=int, default=50_000)
    args.add_argument("--batch-size", type=int, default=10)
    args.add_argument("--output-path", type=str, default="/ibex/ai/home/alyafez/.cache/jql-synthesize")
    args.add_argument("--max-tokens", type=int, default=4096)
    args.add_argument("--overwrite", type=bool, default=False)
    args.add_argument("--shard", type=int, default=0)
    args.add_argument("--num-shards", type=int, default=1)
    args = args.parse_args()
    file_path = glob("/ibex/ai/project/c2254/arabic_filtered_data/fineweb2_arabic_threshold_2/data/arb_Arab/train/compressed/*.parquet")[args.shard % args.num_shards]
    fw = load_dataset("parquet", data_files=[file_path], split = f'train[:{args.num_examples}]')
    file_name = file_path.split("/")[-1].replace(".parquet", ".json")
    print(f"Processing {file_name}")
    prompter = Prompter(model=args.model, max_tokens=args.max_tokens, output_path=args.output_path, overwrite=args.overwrite, num_examples=args.num_examples, file_name = file_name)
    prompter.process_dataset(fw, args.batch_size)

if __name__ == "__main__":
    main()
