import argparse
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
from datasets import load_dataset, concatenate_datasets
import pyarrow.parquet as pq
import os
from glob import glob

def get_total_examples(args):
    total_examples = 0
    files = sorted(glob(args.base_path + "*.parquet"))
    for file in files:
        metadata = pq.read_metadata(file)
        num_examples = metadata.num_rows
        # print(file, num_examples)
        total_examples += num_examples
    return files, total_examples

def get_total_examples_dummy():
    return sum([len(x) for x in dummy_examples])   
        
dummy_examples = [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]]

def get_shard_examples(shard, num_shards):
    files, total_examples = get_total_examples(args)
    max_examples = total_examples // num_shards
    # files = dummy_examples
    curr_total_examples = 0
    fw = None
    for i in range(0, len(files)):
        first_file_num_rows = pq.read_metadata(files[i]).num_rows
        # first_file_num_rows = len(files[i])
        curr_range = (curr_total_examples, curr_total_examples + first_file_num_rows)
        if shard * max_examples >= curr_range[0] and shard * max_examples < curr_range[1]:
            print(i, first_file_num_rows)
            fw1 = load_dataset("parquet", data_files=[files[i]], split = 'train', num_proc=1)
            start_idex = shard * max_examples % (first_file_num_rows)
            end_idex = min(start_idex + max_examples, len(fw1))
            print('start', start_idex)
            print('end', end_idex)
            fw1 = fw1.select(range(start_idex, end_idex))
            fw = fw1
            if len(fw1) < max_examples:
                print('concatenating')
                # print('fw1', len(fw1))
                if i + 1 < len(files):
                    print('adding ', max_examples - len(fw1))
                    fw2 = load_dataset("parquet", data_files=[files[i + 1]], split = 'train', num_proc=1).select(range(0, max_examples - len(fw1)))
                    # print('fw2', len(fw2))
                    fw = concatenate_datasets([fw1, fw2])
            assert len(fw) == max_examples, f"fw length {len(fw)} != max_examples {max_examples}"
            if i == len(files) - 1 and shard == num_shards - 1: # flush last shard
                # print('flushing')
                _fw = load_dataset("parquet", data_files=[files[i]], split = 'train', num_proc=1)
                fw = concatenate_datasets([fw, _fw.select(range(end_idex, len(_fw)))])
        curr_total_examples += first_file_num_rows
    return fw

def get_file_idex(files, shard, max_examples):
    curr_total_examples = 0
    for i in range(0, len(files)):
        first_file_num_rows = pq.read_metadata(files[i]).num_rows
        # first_file_num_rows = len(files[i])
        curr_range = (curr_total_examples, curr_total_examples + first_file_num_rows)
        if shard * max_examples >= curr_range[0] and shard * max_examples < curr_range[1]:
            return curr_total_examples, i
        curr_total_examples += first_file_num_rows
    return None
        
def get_shard_examplesv2(shard, num_shards):
    files, total_examples = get_total_examples(args)
    max_examples = total_examples // num_shards
    # files = dummy_examples
    fw = None
    total_before, i = get_file_idex(files, shard, max_examples)
    num_rows = pq.read_metadata(files[i]).num_rows
    fw1 = load_dataset("parquet", data_files=[files[i]], split = 'train', num_proc=1)
    start_idex = shard * max_examples - total_before
    end_idex = min(start_idex + max_examples, num_rows)
    print(i, start_idex, end_idex)
    fw1 = fw1.select(range(start_idex, end_idex))
    fw = fw1
    if len(fw1) < max_examples:
        if i + 1 < len(files):
            fw2 = load_dataset("parquet", data_files=[files[i + 1]], split = 'train', num_proc=1).select(range(0, max_examples - len(fw1)))
            fw = concatenate_datasets([fw1, fw2])
    assert len(fw) == max_examples, f"fw length {len(fw)} != max_examples {max_examples}"
    if i == len(files) - 1 and shard == num_shards - 1: # flush last shard
        _fw = load_dataset("parquet", data_files=[files[i]], split = 'train', num_proc=1)
        fw = concatenate_datasets([fw, _fw.select(range(end_idex, len(_fw)))])
    return fw
def main(args):
    # total_after_split = 0
    # for i in range(0, args.num_shards):
    #     print(f"Processing shard {i}")
    #     fw = get_shard_examplesv2(i, args.num_shards)
    #     # print(len(fw))
    #     total_after_split += len(fw)
    # print(total_after_split)
    fw = get_shard_examplesv2(args.shard, args.num_shards)
    print(len(fw))

    while True:
        try:
            output_path_template = f"{args.base_path}/compressed/{args.shard:05d}.parquet"
            fw.to_parquet(output_path_template.format(index=args.shard), compression = "brotli")
            break
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/ibex/ai/project/c2254/arabic_filtered_data/fineweb2_arabic_threshold_2/data/arb_Arab/train/")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=2)

    args = parser.parse_args()
    main(args)

