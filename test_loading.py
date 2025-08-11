from datasets import load_dataset
import os
import pyarrow.parquet as pq
from glob import glob
base_path = "/ibex/ai/project/c2254/arabic_filtered_data/fineweb2_arabic_threshold_2/data/arb_Arab/train/compressed/"
def get_total_examples(base_path):
    total_examples = 0
    files = sorted(glob(base_path + "*.parquet"))
    print(files)
    for file in files:
        metadata = pq.read_metadata(file)
        num_examples = metadata.num_rows
        print(file.split("/")[-1], num_examples)
        total_examples += num_examples
    return total_examples   

def main():
    total_examples = get_total_examples(base_path)
    print(total_examples)
    # all_files = glob("/ibex/ai/project/c2254/arabic_filtered_data/fineweb2_arabic_threshold_2/data/arb_Arab/train/compressed/*.parquet")
    # print(sorted(all_files))
    # for i in range(0, 64):
    #     fw = load_dataset("parquet", data_files=[base_path + f"{i:05d}.parquet"], split = 'train', num_proc=1)
    #     print(fw)

if __name__ == "__main__":
    main()
