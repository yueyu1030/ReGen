# ReGen
This is the code repo for our ACL 2023 Findings paper [ReGen: Zero-Shot Text Classification via Training Data Generation with Progressive Dense Retrieval](https://arxiv.org/pdf/2305.10703.pdf).

**Update**: Checkout how to improve ReGen using large language models in our recent [preprint](https://arxiv.org/abs/2306.15895) with [code](https://github.com/yueyu1030/AttrPrompt)!

## Dependencies
```
python 3.8
transformers==4.2.0
pytorch==1.8.0
scikit-learn
faiss-cpu==1.6.4
tqdm>=4.62.2
nltk
```

# Data
### Download Corpus
The corpus can be downloaded at:
- [This link](https://huggingface.co/datasets/yyu/review_corpus) for reviews.
- [This link](https://huggingface.co/datasets/yyu/news_corpus) for news.
- [This link](https://huggingface.co/datasets/yyu/wiki_corpus) for wikipedia.

### Download Data
The test set of {AG News, DBPedia, Yahoo, IMDB} can be easily find at huggingface data hub. The test sets for other datasets can be founded at the `test` folder.



### Data Format
The `_id` stands for the class id, and `text` is the content of the document.

Example (for SST-2 Dataset):
```
{
    {"_id": 0, "text": "It seems to me the film is about the art of ripping people off without ever letting them consciously know you have done so."}
    {"_id": 0, "text": "In the end , the movie collapses on its shaky foundation despite the best efforts of director joe carnahan."}
    {"_id": 1, "text": "Despite its title , punch-drunk love is never heavy-handed ."}
    {"_id": 1, "text": "Though only 60 minutes long , the film is packed with information and impressions."}
    ...
}
```
# Model 
## Contrastive Pretraining Step
We adapt the code from [COCO-DR](https://github.com/OpenMatch/COCO-DR/tree/main/COCO) for pretraining. Please check the original implementation for details. 

**Updated on Sep 7th 2023**: The pretrained model has been released on the HuggingFace:

- News: [yyu/news_contrastive_pretrain](https://huggingface.co/yyu/news_contrastive_pretrain)
- Wiki: [yyu/wiki_contrastive_pretrain](https://huggingface.co/yyu/wiki_contrastive_pretrain)
- Review: [yyu/review_contrastive_pretrain](https://huggingface.co/yyu/review_contrastive_pretrain)

## Retrieval Step
### Embedding Generation
See the code from the  `retrieval` folder, `gen_embedding.sh` for details.

### Retrieval
See the code from  `retrieval/retrieve.py`  for details.

Some Key Hyperparameters:
- `args.target`: The target dataset used in the experiment.
- `args.model`: The retrieval model used in this study.
- `args.corpus_folder/args.corpus_name`: The folder/name of the corpus used (e.g. News, Wiki) in the experiments.
- `args.topN`: The topN used in KNN search (usually set to 50-100).
- `args.round`: The retrieval rounds. Set to 0 for the first rounds (using label name/template for retrieval only) and 1,2,... for later rounds.

**Note**: in principle, our model is compatible with any dense retrievers (after properly training). If you want to use your own dense retrieval model, please make sure that the dense retrieval model also use the embedding of [CLS] token as sequence embeddings. Otherwise, you may need to modify the code in embedding generation parts to make sure the embedding generated is *correct*.

## Classification Step
### Noisy Data Removal
See the code from the `filter` folder. The example command should be
```
train_cmd="CUDA_VISIBLE_DEVICES=0 python3 inference.py --task=${task} \
	--unlabel_file=${unlabel_file_used_for_filtering} \
	--data_dir=${folder_for_data}	\
	--cache_dir="${task}/cache" --output_dir=${output_dir} --round=${round} \
	--load_from_prev=1 \
	--gpu=${gpu}  --eval_batch_size=${eval_batch_size} \
	--max_seq_len=${max_seq_len} --auto_load=0 \
	--model_type=${model_type}"
echo $train_cmd
eval $train_cmd
```
Here
- `folder_for_data` is the folder of the retrieved data.
- `unlabel_file_used_for_filtering` is the file name of the retrieved data.
- `task` is the name of the task.
- `model_type` is the PLM used as the discriminator (e.g. RoBERTa).

### Classifier Training
See the code from the `classification` folder. The example command should be
```
train_cmd="CUDA_VISIBLE_DEVICES=0 python3 main.py --do_train --do_eval --task=${task} \
	--train_file={PATH_FOR_GENERATED_DATASET} \
	--dev_file={PATH_FOR_GENERATED_VALID_DATASET \
	--test_file={PATH_FOR_TEST_DATASET \
	--unlabel_file=unlabeled.json \
	--data_dir=../datasets/${task}-${label_per_class} --train_seed=${train_seed} \
	--cache_dir="../datasets/${task}-${label_per_class}/cache" \
	--output_dir=${output_dir} \
	--logging_steps=${logging_steps} \
	--n_gpu=${n_gpu} --num_train_epochs=6 \
	--learning_rate=2e-5   --weight_decay=1e-8 \
	--batch_size=32 --eval_batch_size=128 \
	--max_seq_len=128 --auto_load=1 \
	--model_type=${model_type}"
echo $train_cmd
eval $train_cmd
```

## Progressive Retrieval
It is achieved with a similar way to the previous retrieval step. 
See the code from  `retrieval/retrieve.py` again for details. 
The only difference is that you need to set the variable `args.round` to greater than `0`. You also need to set the `prev_retrieve_path_name` and `prev_retrieve_folder` to the path of the documents for the latest retrieval results *after filtering*.


## Generated Dataset
The generated dataset can be found at [this Link](https://drive.google.com/drive/folders/1mW91mfNqt5COZcIJg8QMhjMoWjGMyAm-?usp=share_link).

# Reference
Please kindly cite our paper if you find this repo useful for your research. Thanks!
```
@inproceedings{yu2023zero,
  title={ReGen: Zero-Shot Text Classification via Training Data Generation with Progressive Dense Retrieval},
  author={Yu, Yue and Zhuang, Yuchen and Zhang, Rongzhi and Meng, Yu and Shen, Jiaming and Zhang, Chao},
  booktitle={Findings of ACL},
  year={2023}
}
```
