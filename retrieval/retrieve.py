from re import L
import faiss 
import argparse
import json
from tqdm import trange, tqdm
import numpy as np
import pickle
from transformers import AutoModel, AutoTokenizer
import torch 
import os 

## load embedding
def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--dataset",
        default='news_corpus',
        type=str,
        required=True,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--model_name",
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
        default=32,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--gpuid",
        default=0,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--target",
        default='agnews',
        type=str,
        help="The name of the target dataset",
    )

    parser.add_argument(
        "--N",
        default=20,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--round",
        default=1,
        type=int,
        help="The round of retrieval",
    )

    parser.add_argument(
        "--prev_retrieve_path_name",
        default=1,
        type=str,
        help="The name of document for saving the retrieval results from previous rounds",
    )

    parser.add_argument(
        "--prev_retrieve_folder",
        default=1,
        type=str,
        help="The folder for saving the retrieval results from previous rounds",
    )

    parser.add_argument(
        "--corpus_folder",
        default=1,
        type=str,
        help="The folder for save the general domain corpus",
    )


    args = parser.parse_args()
    return args


args = get_arguments()
text = []
label = []

print("Model Name:", args.model_name)

print("Loading Text")
with open(f"{args.corpus_folder}/{args.type}.jsonl", 'r') as f:
    for lines in f:
        lines = json.loads(lines)
        text.append(lines["text"])
    print("corpus size:", len(text),)

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModel.from_pretrained(args.model_name)
model = model.to(f"cuda:{args.gpuid}")
if args.target == 'agnews':
    qtext = [
        "an world politics news.", 
        "a sports news.",
        "a business news.", 
        "a science and technology news."
    ]
    id2label = [0, 1, 2, 3]
elif args.target == 'dbpedia':
    qtext= [
        'business company',
        'school university', 
        'artist', 
        'sports athlete', 
        'politics', 
        'transportation', 
        'building', 
        'river, mountain, lake',
        'village', 
        'an animal', 
        'plant tree', 
        'album', 
        'movie', 
        'novel, publication, book',
    ]
    id2label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
elif args.target == 'nyt':
    qtext= [
        'business news',
        'politics news', 
        'sports news', 
        'health news', 
        'education news', 
        'real estate news', 
        'art news', 
        'science news',
        'technology news', 
    ]
    id2label = [0, 1, 2, 3, 4, 5, 6, 7, 8]
elif args.target == 'yahoo':
    qtext = [
        'society culture',
        'science math', 
        'health', 
        'school education', 
        'computer internet', 
        'sports', 
        'business finance', 
        'music film',
        'family love', 
        'politics government', 
    ]
    id2label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
elif args.target == 'review':
    qtext = [
        'it was a bad film',
        'it was a great film', 
    ]
    id2label = [0, 1]


text_tmp = []

if args.round >= 1:
    id2label = []
   
    path = args.prev_retrieve_path
    print(f"round {args.round}, path {path}")
    with open(path, "r") as f:
        for lines in tqdm(f):
            lines = json.loads(lines)
            id2label.append(lines["_id"])
            text_tmp.append(qtext[lines["_id"]] + " " + tokenizer.sep_token + " " + lines["text"])
    qtext = text_tmp
    id2label = np.array(id2label)
    print(len(qtext), len(id2label))
print("Query Embedding")

q_embeddings = []
num_iter = len(qtext)//args.batch_size if len(qtext) % args.batch_size == 0 else (len(qtext)//args.batch_size + 1)
for i in trange(num_iter):
    inputs = tokenizer(qtext[i*args.batch_size:(i+1)*args.batch_size], max_length = 100 if args.round > 0 else 64, padding=True, truncation=True, return_tensors="pt").to(f"cuda:{args.gpuid}")
    # Get the embeddings
    with torch.no_grad():
        if 'simcse' in args.model:
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        else:
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1]
            embeddings = embeddings.squeeze(1)
        q_embeddings.append(embeddings.cpu().numpy())
q_embeddings = np.concatenate(q_embeddings, axis = 0)

print("Loading Passage Embedding")

with open(f"{args.corpus_folder}/{args.dataset}/embedding_{args.model}_{args.type}.pkl", 'rb') as handle:
    passage_embedding = pickle.load(handle)

print("Calculating FAISS")
dim = q_embeddings.shape[1]
faiss.omp_set_num_threads(32)
cpu_index = faiss.IndexFlatIP(dim)
topN = args.N
cpu_index.add(passage_embedding)    
dev_D, dev_I = cpu_index.search(q_embeddings, topN)
print(dev_I.shape[0])
import os 
os.makedirs(f"{args.corpus_folder}/{args.target}", exist_ok = True)
file_name = f"{args.corpus_folder}/{args.target}/{args.target}_{args.model}_{args.type}_top{topN}_round{args.round}.jsonl" # for saving the retrieved results

visited = {}

if args.round == 0:
    with open(file_name, 'w') as f:
        for i in range(dev_I.shape[0]):
            for j in range(topN):
                data = {"_id": int(i), "text": text[dev_I[i][j]], "docid": int(dev_I[i][j]), "sim": "{:.4f}".format(dev_D[i][j])}
                f.write(json.dumps(data) + '\n')

else:
    with open(file_name, 'w') as f:
        for i in range(dev_I.shape[0]):
            for j in range(topN):
                doc_id = int(dev_I[i][j])
                doc_embedding = passage_embedding[doc_id].reshape(1, -1)
                doc_label = id2label[i]
                dim = q_embeddings.shape[1]
                faiss.omp_set_num_threads(32)
                cpu_index = faiss.IndexFlatIP(dim)
                cpu_index.add(q_embeddings)    
                dev_D_doc, dev_I_doc = cpu_index.search(doc_embedding, 5)
                qknn_id = id2label[dev_I_doc[0]]
                if np.any(qknn_id != doc_label):  # idea close to round-trip filtering
                    continue
                visited[doc_id] = 1    
                data = {"_id": int(id2label[i]), "text": text[doc_id], "docid": doc_id, "sim": "{:.4f}".format(dev_D[i][j])}
                f.write(json.dumps(data) + '\n')
