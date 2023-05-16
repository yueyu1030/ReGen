# ReGen
This is the code repo for our ACL'23 Findings paper "ReGen: Zero-Shot Text Classification via Training Data Generation with Progressive Dense Retrieval".


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

### Download Data

### Data Format

# Model 

# Retrieval Step
## Embedding Generation
See the code from the  `retrieval` folder, `gen_embedding.sh` for details.

## Retrieval
See the code from the  `retrieval` folder, `gen_embedding.sh` for details.

# Classification Step
## Noisy Data Removal
See the code from the `filter` folder. The example command should be
```
train_cmd="CUDA_VISIBLE_DEVICES=0 python3 inference.py --task=${task} \
	--unlabel_file=${unlabel_file_used_for_filtering} \
	--data_dir=${folder_for_data} --seed=${seed} --train_seed=${train_seed} \
	--cache_dir="${task}/cache" --output_dir=${output_dir} --round=${round} --load_from_prev=1 \
	--gpu=${gpu} --n_gpu=${n_gpu} \
	--method=${method} --eval_batch_size=${eval_batch_size} \
	--max_seq_len=${max_seq_len} --auto_load=0 \
	--model_type=${model_type}"
echo $train_cmd
eval $train_cmd
```
## Classifier Training
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

# Progressive Retrieval

# Generated Dataset
The generated dataset can be found at 

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