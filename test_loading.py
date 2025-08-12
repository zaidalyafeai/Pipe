from datasets import load_dataset
from run_compression import get_total_examples
from glob import glob
base_path = "/ibex/ai/project/c2254/arabic_filtered_data/fineweb2_arabic_threshold_2/data/arb_Arab/train/"
compressed_path = base_path + "compressed/"

def main():
    files, total_examples = get_total_examples(base_path)
    compressed_files, total_compressed_examples = get_total_examples(compressed_path)
    print('len of files', len(files))
    print('len of compressed files', len(compressed_files))
    print(total_examples, total_compressed_examples)
    print()    
    fw = load_dataset("parquet", data_files=[files[-1]], split = 'train', num_proc=1)
    print(fw[-1])
    fw = load_dataset("parquet", data_files=[compressed_files[-1]], split = 'train', num_proc=1)
    print(fw[-1])

if __name__ == "__main__":
    main()
