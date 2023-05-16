import os
import logging
from tqdm import tqdm, trange
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset, TensorDataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoModelForSequenceClassification
import copy
import math
import os
import random 
from sklearn.metrics import f1_score
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from collections import Counter
from losses import SCELoss, LabelSmoothingLoss, GCELoss
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0  and torch.cuda.is_available():
        # print('yes')
        # assert 0
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def acc_and_f1(preds, labels, average='macro'):
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average='macro')
    #macro_recall = recall_score(y_true=labels, y_pred = preds, average = 'macro')
    #micro_recall = recall_score(y_true=labels, y_pred = preds, average = 'micro')
    #print(acc, macro_recall, micro_recall)

    return {
        "acc": acc,
        "f1": f1
    }

class Trainer(object):
    def __init__(self, args, train_dataset = None, dev_dataset = None, test_dataset = None, unlabeled = None, \
                num_labels = 10, data_size = 100, n_gpu = 1):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.unlabeled = unlabeled
        self.data_size = data_size

        self.num_labels = num_labels
        self.config_class = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=self.num_labels)
        self.n_gpu = 1
        
    def soft_frequency(self, logits, soft = True):
        """
        Unsupervised Deep Embedding for Clustering Analysis
        https://arxiv.org/abs/1511.06335
        """
        power = self.args.self_training_power
        y = logits
        f = torch.sum(y, dim=0)
        t = y**power / f
        #print('t', t)
        t = t + 1e-10
        p = t/torch.sum(t, dim=-1, keepdim=True)
        return p if soft else torch.argmax(p, dim=1)

    def reinit(self):
        self.load_model()
        self.init_model()

    def init_model(self):
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and self.n_gpu > 0 else "cpu"
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
    
    def load_model(self, path = None):
        print("load Model")
        if path is None:
            logger.info("No ckpt path, load from original ckpt!")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name_or_path,
                config=self.config_class,
                cache_dir=self.args.cache_dir if self.args.cache_dir else None,
            )
        else:
            print(f"Loading from {path}!")
            logger.info(f"Loading from {path}!")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path,
                config=self.config_class,
                cache_dir=self.args.cache_dir if self.args.cache_dir else None,
            )


    def save_model(self, stage = 0):
        output_dir = os.path.join(
            self.args.output_dir,   "checkpoint-{}".format(len(self.train_dataset)), "iter-{}".format(stage), f"seed{self.args.train_seed}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

    def evaluate(self, mode, dataset = None, global_step=-1, return_preds = False):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        # logger.info("  Batch size = %d", self.args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                          }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }
        preds_probs = np.exp(preds)/np.sum(np.exp(preds), axis =-1, keepdims = True)
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        result.update(result)
        logger.info("***** Eval results *****")
        if return_preds:
            print("=================")
            print("Confusion Matrix:")
            print(confusion_matrix(out_label_ids, preds))
            print("====================")
            return results["loss"], result["acc"], result["f1"], preds_probs, out_label_ids
        else:
            return results["loss"], result["acc"], result["f1"]
    def inference(self, layer = -1, train_only = False):
        ## Inference the embeddings/predictions for unlabeled data
        train_dataloader = DataLoader(self.train_dataset, shuffle=False, batch_size=self.args.eval_batch_size)
        train_pred = []
        train_feat = []
        train_label = []
        self.model.eval()
        softmax = nn.Softmax(dim = 1)
        for batch in tqdm(train_dataloader, desc="Evaluating Labeled Set"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                            'output_hidden_states': True
                          }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits, feats = outputs[0], outputs[1], outputs[2]
                logits = softmax(logits).detach().cpu().numpy()
                train_pred.append(logits)
                train_feat.append(feats[layer][:, 0, :].detach().cpu().numpy())
                train_label.append(batch[3].detach().cpu().numpy())
        train_pred = np.concatenate(train_pred, axis = 0)
        train_feat = np.concatenate(train_feat, axis = 0)
        train_label = np.concatenate(train_label, axis = 0)
        train_conf = np.amax(train_pred, axis = 1)
        print("train size:", train_pred.shape, train_feat.shape, train_label.shape, train_conf.shape)
        if train_only:
            return train_pred, train_feat, train_label
        unlabeled_dataloader = DataLoader(self.unlabeled, shuffle=False, batch_size=self.args.eval_batch_size)
        unlabeled_pred = []
        unlabeled_logits = []
        unlabeled_feat = []
        unlabeled_label = []
        self.model.eval()
        for batch in tqdm(unlabeled_dataloader, desc="Evaluating Unlabeled Set"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                            'output_hidden_states': True
                          }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits, feats = outputs[0], outputs[1], outputs[2]
                unlabeled_logits.append(logits.detach().cpu().numpy())
                logits = softmax(logits).detach().cpu().numpy()
                unlabeled_pred.append(logits)
                unlabeled_feat.append(feats[layer][:, 0, :].detach().cpu().numpy())
                unlabeled_label.append(batch[3].detach().cpu().numpy())
        unlabeled_feat = np.concatenate(unlabeled_feat, axis = 0)
        unlabeled_label = np.concatenate(unlabeled_label, axis = 0)
        unlabeled_pred = np.concatenate(unlabeled_pred, axis = 0)
        unlabeled_logits = np.concatenate(unlabeled_logits, axis = 0)
        unlabeled_conf = np.amax(unlabeled_pred, axis = 1)
        unlabeled_pseudo = np.argmax(unlabeled_pred, axis = 1)
        
        print("unlabeled size:", unlabeled_pred.shape, unlabeled_feat.shape, unlabeled_label.shape, unlabeled_conf.shape)
        return train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo

    def train(self, n_sample = 20):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        training_steps = max(self.args.max_steps, int(self.args.num_train_epochs) * len(train_dataloader))

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", training_steps)
        global_step = 0
        tr_loss = 0.0

        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args)
        criterion = LabelSmoothingLoss(num_classes = self.num_labels, smoothing = 0.1, dim = -1, weight = None)
        best_model = None
        best_dev = -np.float('inf')
        best_test =  -np.float('inf')
        for _ in train_iterator:
            global_step = 0
            tr_loss = 0.0
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            local_step = 0
            training_len = len(epoch_iterator)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                          }
                outputs = self.model(**inputs)
                logits = outputs[1]
                
                loss = criterion(pred = logits, target = batch[3].to(self.device))
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps           
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    local_step += 1
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    self.model.zero_grad()
                    global_step += 1
                    epoch_iterator.set_description("iteration:%d, Loss:%.3f, best dev:%.3f" % (_, tr_loss/global_step, 100*best_dev))
                    if self.args.logging_steps > 0 and local_step in [training_len//4, training_len//2]: #   training_len//4,  training_len//4, training_len//2, 3*training_len//4  [100, 200, 300, 400, 500, 600]: # , ] and global_step % self.args.logging_steps == 0:
                        loss_dev, acc_dev, f1_dev = self.evaluate('dev', global_step)
                        print("GLOBAL STEP", global_step, 'acc_dev:', acc_dev, 'f1', f1_dev)
                        if acc_dev > best_dev:
                            logger.info("Best model updated!")
                            self.best_model = copy.deepcopy(self.model.state_dict())
                            best_dev = acc_dev            

                if 0 < training_steps < global_step:
                    epoch_iterator.close()
                    break
            loss_dev, acc_dev, f1_dev = self.evaluate('dev', global_step)
            
            loss_test, acc_test, f1_test = 0 ,0, 0
            if acc_dev > best_dev:
                logger.info("Best model updated!")
                self.best_model = copy.deepcopy(self.model.state_dict())
                best_dev = acc_dev
            print(f'Dev: Loss: {loss_dev}, Acc: {acc_dev}, F1: {f1_dev}', f'Test: Loss: {loss_test}, Acc: {acc_test}, F1: {f1_test}')
        loss_test, acc_test, acc_f1, preds_probs, out_label_ids = self.evaluate('test', global_step, return_preds = True)
        print(f'Test: Loss: {loss_test}, Acc: {acc_test}, F1: {acc_f1}')
        self.save_model(stage = n_sample)
        return global_step, tr_loss / global_step
    
