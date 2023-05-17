
import argparse
import os
from utils import load_and_cache_unlabeled_examples, init_logger, load_tokenizer
from trainer import Trainer
import torch 
import numpy as np 
import random 
import torch.nn as nn
import json
import pickle 

def save_data( unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo, dataset = 'agnews', n_iter = 0):
    # if n_iter == 0:
    #   /  path = f"/localscratch/yueyu/datasets/{dataset}_openws/pred/unlabeled/round_{n_iter}_prev"
        # 
    # else:
    path = f"/localscratch/yueyu/datasets/{dataset}_openws/pred/unlabeled/round{n_iter}_prev"
    os.makedirs(path, exist_ok = True)
    
  
    with open(f"{path}/unlabeled_pred.npy", 'wb') as f:
        np.save(f, unlabeled_pred)

    with open(f"{path}/unlabeled_feat.npy", 'wb') as f:
        np.save(f, unlabeled_feat)
    
    with open(f"{path}/unlabeled_label.npy", 'wb') as f:
        np.save(f, unlabeled_label)
    
    with open(f"{path}/unlabeled_pseudo.npy", 'wb') as f:
        np.save(f, unlabeled_pseudo)


def load_data(dataset = 'SST-2', embedding_model = 'roberta-base', template_id = 0):
    path = f'/localscratch/yueyu/datasets/{dataset}-0-0/'
    with open(path + f'embedding_{embedding_model}_roberta.pkl', 'rb') as f:
        train_emb = pickle.load(f)
    with open(path + f'embedding_{embedding_model}_roberta_tsne.pkl', 'rb') as f:
        train_emb_tsne = pickle.load(f)
    test_emb = []
    # with open(path + f'embedding_{embedding_model}_roberta_test.pkl', 'rb') as f:
    #     test_emb = pickle.load(f)
    
    train_prompt_pred = np.load(path + f"pred_unlabeled_roberta-base_temp{template_id}.npy")
    try:
        train_prompt_logit = np.load(path + f"logit_unlabeled_roberta-base_temp{template_id}.npy")
    except:
        train_prompt_logit = []
    train_label = np.load(path + "pred_labels.npy")

    # train_emb = 
    test_label = []
    if dataset == 'mnli':
        test_file = 'test_m.json'
    else:
        test_file = 'test.json'
    with open(path + test_file, 'r') as f:
        for lines in f:
            test_file = json.loads(lines)
            test_label.append(int(test_file["_id"])) 
    # assert len(test_label) == test_emb.shape[0]
    assert train_emb.shape[0] == train_label.shape[0]
    assert train_emb.shape[0] == train_prompt_pred.shape[0]
    return train_emb, train_emb_tsne, train_prompt_pred, train_prompt_logit, train_label, test_emb, test_label


def set_seed(args):
    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if args.n_gpu > 0  and torch.cuda.is_available():
        # print('yes')
        # assert 0
        torch.cuda.manual_seed_all(args.train_seed)
        torch.cuda.manual_seed(args.train_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)
    if args.task in ['yahoo']:
        n_classes = 10
    elif args.task in ['agnews']:
        n_classes = 4
    elif args.task in ['dbpedia']:
        n_classes = 14
    elif args.task in ['imdb', 'amazon', 'amazon-polarity']:
        n_classes = 2
    elif args.task in ['nyt']:
        n_classes = 9

    unlabeled_dataset, unlabeled_size = load_and_cache_unlabeled_examples(args, tokenizer, mode = 'unlabeled')
    print('unlabel_size:', unlabeled_size)

    trainer = Trainer(args, unlabeled = unlabeled_dataset, num_labels = n_classes)
    trainer.init_model()

    unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo = trainer.inference()

    save_data(unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo, dataset = args.task, n_iter = args.round)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default='0,1,2,3', type=str, help="which gpu to use")
    parser.add_argument("--n_gpu", default=1, type=int, help="which gpu to use")

    parser.add_argument("--seed", default=0, type=int, help="which seed to use")
    parser.add_argument("--train_seed", default=0, type=int, help="which seed to use")
    parser.add_argument("--task", default="agnews", type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="../datasets", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    parser.add_argument("--eval_dir", default="./eval", type=str, help="Evaluation script, result directory")
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--dev_file", default="dev.tsv", type=str, help="dev file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--unlabel_file", default="unlabeled.tsv", type=str, help="Test file")
    parser.add_argument("--do_extra_eval", action="store_true", help="Whether to run extra eval on the test set.")

    parser.add_argument("--extra_dataset", default="", type=str, help="Whether to run extra eval on the test set.")

    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.",)

    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X updates steps.")
    parser.add_argument('--self_train_logging_steps', type=int, default=20, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")
    parser.add_argument("--model_type", default="bert-base-uncased", type=str)
    parser.add_argument("--load_from_prev", default=0, type=int)

    parser.add_argument("--auto_load", default=1, type=int, help="Auto loading the model or not")
    parser.add_argument("--add_sep_token", action="store_true", help="Add [SEP] token at the end of the sentence")

    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=100, type=int, help="Training steps for initialization.")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--eval_batch_size", default=256, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--max_seq_len_test", default=128, type=int, help="The maximum total input sequence length after tokenization.")

    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate_st", default=1e-5, type=float, help="The initial learning rate for Adam.")

    parser.add_argument("--sample_labels", default=32, type=int, help="The initial learning rate for Adam.")

    parser.add_argument('--k_cal', type = int, default = 5, help = 'coefficient of the virtual adversarial training loss term.')

    parser.add_argument('--label_per_class', type = int, default = 10, help = 'virtual adversarial training.')
    parser.add_argument('--num_unlabeled', type = int, default = 100, help = 'virtual adversarial training.')
    parser.add_argument('--num_unlabeled_add', type = int, default = 100, help = 'virtual adversarial training.')

    parser.add_argument('--dr_N', type = int, default = 10, help = 'virtual adversarial training.')

    parser.add_argument('--filter', type = int, default = 0, help = 'Whether use filter.')
    parser.add_argument('--template', type = int, default = 0, help = 'Prompt Template.')
    parser.add_argument('--round', type = int, default = 0, help = 'Prompt Template.')
    parser.add_argument('--word_template', type = int, default = 0, help = 'Prompt Template.')
    parser.add_argument('--n_labels', type = int, default = 0, help = 'Prompt Template.')

# semi_method
    args = parser.parse_args()
    
    args.model_name_or_path = args.model_type
    if args.load_from_prev:
        args.tokenizer = 'roberta-base'
    else:
        args.tokenizer = args.model_name_or_path

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    main(args)