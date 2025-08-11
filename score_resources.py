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



load_dotenv("../../.env")
SAMPLE_RESPONSE = """
```json
{"score": 2, "reasoning": "The extract is a complex political analysis of the Palestinian leadership crisis, the role of the Palestinian Authority, and the ongoing Israeli occupation. It delves into the failures of the Oslo Accords and the challenges of achieving genuine self-determination for Palestinians. While the content is intellectually stimulating, it is far too complex and politically charged for primary or even most grade school students. The language is sophisticated, and the concepts require a significant understanding of Middle Eastern history and politics. It's more suited for a high school or university-level political science course. The text doesn't offer any educational exercises or simplified explanations suitable for younger learners. It's a well-written piece of political commentary, but not a readily usable educational resource for the target age groups. It doesn't contain any harmful content, but its complexity and focus on a sensitive political issue limit its educational value for younger students.", "keywords": ["Palestinian leadership", "Israeli occupation", "Oslo Accords", "Palestinian Authority", "Hamas", "Palestinian elections", "Right of return", "BDS movement", "Political reform", "Self-determination"], "thinking": ""}
```
"""
SYSTEM_PROMPT = """
You are an AI assistant that classifies papers into multiple categories using the title and abstract of the paper.
You should predict if the paper introduces or releases a new dataset or benchmark for nlp, computer vision or speech.
The abstract MUST explicitly mention the creation of a new dataset. Do NOT make any assumptions.
Each dataset paper must be classified in one of the following categories:
- ar: the paper introduces a new  dataset in Arabic or its dialects
- en: the paper introduces a new dataset in English, if the language is not mentioned then assume it is English.
- fr: the paper introduces a new dataset in French
- ru: the paper introduces a new dataset in Russian
- jp: the paper introduces a new dataset in Japanese
- other: the paper introduces a new dataset in another language not in the list above
- multi: the paper introduces a new multilingual or cross-lingual dataset
If the paper is doesn't belong to any of the categories above, the category should be "none".
The output should be a JSON object with the following fields:
{
    "reasoning": "reasoning why the paper is in the category",
    "category": "one item from the list [ar, en, fr, ru, jp , other, multi, none]"
}
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
    if "<think>" in response:
        thinking = response.split("<think>")[1].split("</think>")[0].strip()
        response = response.split("</think>")[1].strip()
    else:
        thinking = ""
        # raise ValueError("No thinking found in the response")
    response = response.replace("```json", "").strip()
    response = response.replace("```", "").strip()
    return response, thinking

class Prompter():
    def __init__(self, model: str, max_tokens: int = 4096, language: str = "arb_Arab", HOST: str = "localhost", PORT: int = 8787, output_path: str = "/ibex/ai/home/alyafez/.cache/curator", overwrite: bool = False, num_examples: int = 50_000):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(f"google/{model}")
        self.SYSTEM_PROMPT_TOKENS = len(self.tokenizer.encode(SYSTEM_PROMPT)) + 21 # 21 is the number of tokens in the system prompt
        self.SAMPLE_RESPONSE_TOKENS = len(self.tokenizer.encode(SAMPLE_RESPONSE))
        self.max_length = max_tokens - self.SYSTEM_PROMPT_TOKENS - self.SAMPLE_RESPONSE_TOKENS
        print(f"MAX_TOKENS: {max_tokens}")
        print(f"SYSTEM_PROMPT_TOKENS: {self.SYSTEM_PROMPT_TOKENS}")
        print(f"SAMPLE_RESPONSE_TOKENS: {self.SAMPLE_RESPONSE_TOKENS}")
        self.language = language
        self.HOST = HOST
        self.PORT = PORT
        self.client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=f"http://{HOST}:{PORT}/v1",
        )
        self.output_folder = os.path.join(output_path, self.generate_hash_filename())
        os.makedirs(self.output_folder, exist_ok=True)
        self.output_file = os.path.join(self.output_folder, "results.jsonl")
        print(f"model: {model}", f"language: {language}", f"output_path: {self.output_file}")

        if not os.path.exists(self.output_file) or overwrite:
            with open(self.output_file, 'w') as f:
                f.write("")
        self.num_examples = num_examples

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
        fw = fw.filter(lambda x: x["title"] is not None and x["abstract"] is not None)
        fw = fw.filter(lambda x: int(x["year"]) > 2010)
        fw = fw.map(lambda x: {"text": x["title"] + " " + x["abstract"]}, batched=False)
        fw = fw.map(self.truncate, batched=True, batch_size=1000)['train']
        if os.path.exists(self.output_file):
            # load the ids from the file
            print(f"Loading {self.output_file}")
            with open(self.output_file, 'r') as f:
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
        for i in pbar:
            total_requests = 0
            total_tokens = 0
            if i+ batch_size > len(fw):
                batch_size = len(fw) - i
            batch = fw.select(range(i, i+batch_size))
            start_time = time.time()
            batch_results = asyncio.run(self.process_batch(batch))
            results.extend(batch_results)
            
            for example in batch:
                total_tokens += example["num_tokens"]

            # Update RPM metrics
            total_requests += len(batch)
            elapsed_time = time.time() - start_time
            rps = total_requests / elapsed_time  if elapsed_time > 0 else 0
            tps = total_tokens / elapsed_time if elapsed_time > 0 else 0
            rps_list.append(rps)
            tps_list.append(tps)
            rps = sum(rps_list) / len(rps_list)
            tps = sum(tps_list) / len(tps_list)
            # Update progress bar description
            pbar.set_description(f"Processed: {i+len(batch)}/{len(fw)} | RPS: {rps:.2f} | TPS: {tps:.2f}")
            
            with open(self.output_file, 'a') as f:
                for result in batch_results:
                    f.write(json.dumps(result) + '\n')

    async def process_single_example(self, example):
        """Process a single example with retries."""
        max_tries = 0
        output = {
                "id": example["id"],
                "title": example["title"],
                "abstract": example["abstract"],
                "url": example["url"],
                "reasoning": "",
                "category": ""
            }
        while max_tries < 1:
            # Generate prompt
            prompt_text = self.prompt(example)
            
            try:
                # Call OpenAI API
                response = await self.client.chat.completions.create(
                    model=self.model,  
                    messages=[
                        {"role": "system", "content": "You are an AI assistant that evaluates resource papers."},
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=0.1
                )
                response = response.choices[0].message.content
            
                result = self.parse(response)
                output = output | result
                break
            except Exception as e:
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
        return SYSTEM_PROMPT + f"Title: {input['title']}\nAbstract: {input['abstract']}"

    def parse(self,response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        response, thinking = process_json(response)
        response = json.loads(response)
        return response
            

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--mode", type=str, default="vllm")
    args.add_argument("--model", type=str, default="gemma-3-4b-it")
    args.add_argument("--language", type=str, default="arb_Arab")
    args.add_argument("--num-examples", type=int, default=50_000)
    args.add_argument("--batch-size", type=int, default=10)
    args.add_argument("--output-path", type=str, default="/ibex/ai/home/alyafez/.cache/jql")
    args.add_argument("--max-tokens", type=int, default=4096)
    args.add_argument("--overwrite", type=bool, default=False)
    args = args.parse_args()

    prompter = Prompter(model=args.model, max_tokens=args.max_tokens, 
                        language=args.language, output_path=args.output_path, overwrite=args.overwrite, num_examples=args.num_examples)
    fw = load_dataset("csv", data_files="/ibex/ai/home/alyafez/MOLE/arxiv_papers.csv")
    print(fw)
    prompter.process_dataset(fw, args.batch_size)

if __name__ == "__main__":
    main()
