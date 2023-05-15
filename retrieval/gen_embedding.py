import torch
from transformers import AutoModel, AutoTokenizer
import argparse
import json
from tqdm import trange
import numpy as np
import pickle
import os 
# Tokenize input texts
def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--dataset",
        default='agnews',
        type=str,
        required=True,
        help="The input data dir. Should contain the cached passage and query files",
    )
    parser.add_argument(
        "--model",
        default='simcse',
        type=str,
        required=True,
        help="The input data dir. Should contain the cached passage and query files",
    )
    parser.add_argument(
        "--type",
        default='unlabeled',
        type=str,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--gpuid",
        default=0,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
    )
    args = parser.parse_args()
    return args

args = get_arguments()
text = []
label = []
if args.model == 'simcse':
    args.model_name = "princeton-nlp/unsup-simcse-roberta-base"
else:
    args.model_name = args.model 


print("Model Name:", args.model_name)


with open(f"../datasets/{args.dataset}/{args.type}.jsonl", 'r') as f:
    for lines in f:
        lines = json.loads(lines)
        text.append(lines["text"])
    print("size of the corpus", len(text),)

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModel.from_pretrained(args.model_name)

model = model.to(f"cuda:{args.gpuid}")

embedding = []

num_iter = len(text)//args.batch_size if len(text) % args.batch_size == 0 else (len(text)//args.batch_size + 1)
for i in trange(len(text)//args.batch_size + 1):
    inputs = tokenizer(text[i*args.batch_size:(i+1)*args.batch_size], padding=True, truncation=True, max_length = 200, return_tensors="pt").to(f"cuda:{args.gpuid}")

    with torch.no_grad():
        if 'simcse' in args.model:
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        else:
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1]
            embeddings = embeddings.squeeze(1)
        embedding.append(embeddings.cpu().numpy())
    
embedding = np.concatenate(embedding, axis = 0)
print("Embedding Shape:", embedding.shape)

os.makedirs(f"./datasets/{args.dataset}/", exist_ok= True)
with open(f"./datasets/{args.dataset}/embedding_{args.model}_{args.type}.pkl", 'wb') as handle:
    pickle.dump(embedding, handle, protocol=4)


