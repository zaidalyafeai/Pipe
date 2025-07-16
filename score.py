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
SYSTEM_PROMPT = """Below is an extract from a web page in 'Arabic'. Evaluate whether the page has a high educational
value and could be useful in an educational setting for teaching from primary school to
grade school levels for Arabic speakers. Use an additive 5-point scoring system described below. Points are
accumulated based on the satisfaction of each criterion:
- score 0 if the extract is not coherent, contains mixed unrelated topics, or is not written in a way that is easy to understand for Arabic learners.
Give a score of 0 if the extract contains sensitive, explicit, or sexual content. Additionally,
  score 0 if the extract has content topics related to harmful content like drugs, suicide, violence, etc.
  Also score 0 if the extract is a promotional material for a product or service. Do NOT add any additional points for such contents.
- Add 1 point if the extract provides some basic information relevant to educational top-
ics, even if it includes some irrelevant or non-academic content like advertisements and
promotional material.
- Add another point if the extract addresses certain elements pertinent to education but
does not align closely with educational standards. It might mix educational content with
non-educational material, offering a superficial overview of potentially useful topics, or
presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key
concepts relevant to school curricula. It is coherent though it may not be comprehensive
or could include some extraneous information. It may resemble an introductory section of
a textbook or a basic tutorial that is suitable for learning but has notable limitations like
treating concepts that are too complex for grade school students.
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes
for a level not higher than grade school, exhibiting a clear and consistent writing style. It
could be similar to a chapter from a textbook or a tutorial, offering substantial educational
content, including exercises and solutions, with minimal irrelevant information, and the
concepts aren't too advanced for grade school students. The content is coherent, focused,
and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for
teaching either at primary school or grade school. It follows detailed reasoning, the writing
style is easy to follow and offers profound and thorough insights into the subject matter,
devoid of any non-educational or complex content.
After examining the extract:
- Generate a maximum of 5 keywords that are best to describe the topic of the extract. The keywords MUST be in English.
- Reason in less than 5 sentences about what score to give to the extract based on the criteria.
- Output the score using the format: "score: <total points>"
Ensure the output is valid JSON as it will be parsed using `json.loads()` in Python. 
It should be in the following schema, don't add any extra text or json headers: 
{
    "keywords": <keywords>,
    "reasoning": <reasoning>,
    "score": <total points>
}
Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. 
To evaluate the response in alignment with this additive scoring model, we'll systematically attribute points based on the outlined criteria.
The extract is: 
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
        fw = self.create_dataset(fw)
        fw = fw.map(self.truncate, batched=True, batch_size=1000)
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
                "score": 0,
                "reasoning": "",
                "keywords": "",
                "thinking": "",
                "id": example["id"],
                "url": example["url"],
                "input_text": example["text"],
                "model": self.model,
                "language": self.language
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
        """Parse the model response along with the input to the model into the desired output format.."""
        response, thinking = process_json(response)
        response = json.loads(response)
        response["thinking"] = thinking
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
    fw = load_dataset("/ibex/ai/project/c2254/arabic_raw_data/fineweb2_arabic/data/arb_Arab/train", split = 'train', streaming=True)
    prompter.process_dataset(fw, args.batch_size)

if __name__ == "__main__":
    main()
