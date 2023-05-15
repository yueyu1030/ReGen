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

## Classifier Training


# Progressive Retrieval



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