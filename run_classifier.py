import torch
import argparse
from datasets import load_dataset
from utils import load_classifier, compute_scores

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder, regression_head = load_classifier(f'{args.model_name}/regression_head.ckpt', device)

    dataset = load_dataset(f'{args.dataset_name}/{args.dataset_config}/{args.dataset_split}', split = 'train')
    dataset = dataset.shard(index=args.shard, num_shards=args.num_shards, contiguous=True)

    print("length is ", len(dataset), 'per shard number', args.shard)

    dataset = dataset.map(compute_scores, fn_kwargs={'embedder': embedder, 'regression_head': regression_head, 'name': args.model_name}, batched=True, batch_size=6144)
    dataset = dataset.filter(lambda x: x['score'] >= args.threshold)
    print("length is ", len(dataset), 'per shard number', args.shard)

    while True:
        try:
            output_path_template = f"{args.output_dataset_name}/{args.shard:05d}.parquet"
            dataset.to_parquet(output_path_template.format(index=args.shard))
            break
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, default="UBC-NLP__ARBERT@gemma-3-27b-it"
    )
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--dataset_config", type=str, default="arb_Arab")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_name", type=str, default='/ibex/ai/project/c2254/arabic_raw_data/fineweb2_arabic/data')
    parser.add_argument(
        "--output_dataset_name", type=str, default="/ibex/ai/project/c2254/arabic_raw_data/filtered_fineweb2_arabic"
    )
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)

    args = parser.parse_args()
    args.output_dataset_name = f"{args.output_dataset_name}_threshold_{args.threshold}/data/{args.dataset_config}/{args.dataset_split}"
    main(args)