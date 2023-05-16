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



class SCELoss(torch.nn.Module):
    def __init__(self, alpha= 0.1, beta = 1, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # CCE
        ce = self.cross_entropy(pred, target)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.0, dim=-1, weight = None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = num_classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            # print(true_dist)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class GCELoss(nn.Module):
    def __init__(self, thresh = 0.5, q = 0.7, dim=-1, weight = None):
        super(GCELoss, self).__init__()
        self.thresh = thresh
        self.weight = weight
        self.dim = dim
        self.gce_loss_q = q
    
    def forward(self, pred, target):
        input = pred
        # print(input, target)
        # softmax = nn.Softmax(dim=1)
        # target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
        # weight = torch.max(target, axis = 1).values
        # target = torch.argmax(target, dim = -1)
        if self.gce_loss_q == 0:
            if input.size(-1) == 1:
                ce_loss = nn.BCEWithLogitsLoss(reduction='none')
                loss = ce_loss(input.view(-1), input.float())
            else:
                ce_loss = nn.CrossEntropyLoss(reduction='none')
                loss = ce_loss(input, target)
        else:
            if input.size(-1) == 1:
                pred = torch.sigmoid(input)
                pred = torch.cat((1-pred, pred), dim=-1)
            else:
                pred = F.softmax(input, dim=-1)
            pred_ = torch.gather(pred, dim=-1, index=torch.unsqueeze(target, -1))
            w = pred_ > self.thresh
            loss = (1 - pred_ ** self.gce_loss_q) / self.gce_loss_q
            # print(pred_, w)
            loss = loss[w].mean()    
        return loss


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


def compute_metrics_rel(key, prediction):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == 0 and guess == 0:
            pass
        elif gold == 0 and guess != 0:
            guessed_by_relation[guess] += 1
        elif gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        elif gold != 0 and guess != 0:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return {'p':prec_micro, 'r':recall_micro, 'f':f1_micro}


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
    def __init__(self, args, unlabeled = None, contra_datasets= [], \
                num_labels = 10):
        self.args = args
        self.unlabeled = unlabeled
        self.contra_datasets = contra_datasets


        self.num_labels = num_labels
        self.config_class = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=self.num_labels)
        self.n_gpu = 1
        # self.devices = "cuda"
        
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

    def gce_loss(self, input, target, thresh = 0.5, soft = True, conf = None, is_prob = False):
        softmax = nn.Softmax(dim=1)
        if not is_prob:
            target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
            # batch * n_classes
        weight = torch.max(target, axis = 1).values
        target = torch.argmax(target, dim = -1)
        if self.args.gce_loss_q == 0:
            if input.size(-1) == 1:
                ce_loss = nn.BCEWithLogitsLoss(reduction='none')
                loss = ce_loss(input.view(-1), input.float())
            else:
                ce_loss = nn.CrossEntropyLoss(reduction='none')
                loss = ce_loss(input, target)
        else:
            if input.size(-1) == 1:
                pred = torch.sigmoid(input)
                pred = torch.cat((1-pred, pred), dim=-1)
            else:
                pred = F.softmax(input, dim=-1)
            pred_ = torch.gather(pred, dim=-1, index=torch.unsqueeze(target, -1))
            w = pred_ > thresh
            loss = (1 - pred_ ** self.args.gce_loss_q) / self.args.gce_loss_q
            loss = (loss[w])    

        # loss = (loss.view(-1)*weights).sum() / weights.sum()
        return loss
        
    def calc_loss(self, input, target, loss, thresh = 0.5, soft = True, conf = None):
        softmax = nn.Softmax(dim=1)
        target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
        
        if conf == 'max':
            weight = torch.max(target, axis = 1).values
            w = torch.FloatTensor([1 if x == True else 0 for x in weight>thresh]).to(target.device)
        elif conf == 'entropy':
            weight = torch.sum(-torch.log(target+1e-6) * target, dim = 1)
            weight = 1 - weight / np.log(weight.size(-1))
            w = torch.FloatTensor([1 if x == True else 0 for x in weight>thresh]).to(target.device)
        elif conf is None:
            weight = torch.ones(target.shape[0]).to(target.device)
            w =  torch.ones(target.shape[0]).to(target.device)
            
        target = self.soft_frequency(target, soft = soft)
        loss_batch = loss(input, target)
        # print(input, target)
        l = torch.mean(loss_batch * w.unsqueeze(1) * weight.unsqueeze(1))
        # print(weight, w, l)
        confreg = 0.00
        n_classes_ = input.shape[-1]
        l -= confreg *( torch.sum(input * w.unsqueeze(1)) + np.log(n_classes_) * n_classes_ )
        return l

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

    def save_prediction(self, loss, preds, labels, test_preds, test_labels):
        output_dir = os.path.join(
            self.args.output_dir,
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f"{output_dir}/train_pred.npy", 'wb') as f:
            np.save(f, preds)
    
        with open(f"{output_dir}/train_loss.npy", 'wb') as f:
            np.save(f, loss)
        
        with open(f"{output_dir}/train_label.npy", 'wb') as f:
            np.save(f, labels)

        with open(f"{output_dir}/test_label.npy", 'wb') as f:
            np.save(f, test_labels)
        
        with open(f"{output_dir}/test_pred.npy", 'wb') as f:
            np.save(f, test_preds)


    def save_model(self, stage = 0):
        # {self.args.model_type}_{self.args.al_method}
        output_dir = os.path.join(
            self.args.output_dir,  f"{self.args.semi_method}", "checkpoint-{}".format(len(self.train_dataset)), "iter-{}".format(stage), f"seed{self.args.train_seed}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        # tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        # torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
        logger.info("Saving model checkpoint to %s", output_dir)
  
  
    def evaluate(self, mode, dataset = None, global_step=-1, return_preds = False):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'contra':
            dataset = dataset
        elif mode == 'unlabeled':
            dataset = self.unlabeled
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
        if mode == 'unlabeled':
            return preds, preds_probs, out_label_ids
        if 'tacred' in self.args.task:
            result = compute_metrics_rel(preds, out_label_ids)
            result.update(result)
            logger.info("***** Eval results *****")

            # print('Accu: %.4f'%(result["acc"]))
      
            return results["loss"], result['f']
        else:
            result = compute_metrics(preds, out_label_ids)
            result.update(result)
            logger.info("***** Eval results *****")

            # print('Accu: %.4f'%(result["acc"]))
            if return_preds:
                return results["loss"], result["acc"], result["f1"], preds_probs, out_label_ids
            else:
                return results["loss"], result["acc"], result["f1"]
    def inference(self, layer = -1, train_only = False):
        ## Inference the embeddings/predictions for unlabeled data
        softmax = nn.Softmax(dim = 1)
        self.model.eval()
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
        return unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo

    def inference_dataset(self, dataset, layer = -1):
        ## Inference the embeddings/predictions for unlabeled data
        train_dataloader = DataLoader(dataset, shuffle=False, batch_size=self.args.eval_batch_size)
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
                # print(outputs)
                logits = softmax(logits).detach().cpu().numpy()
                train_pred.append(logits)
                train_feat.append(feats[layer][:, 0, :].detach().cpu().numpy())
                train_label.append(batch[3].detach().cpu().numpy())
        train_pred = np.concatenate(train_pred, axis = 0)
        train_feat = np.concatenate(train_feat, axis = 0)
        train_label = np.concatenate(train_label, axis = 0)
        acc = np.mean(np.argmax(train_pred, axis = -1) == train_label) 
        print("train size:", train_pred.shape, train_feat.shape, train_label.shape)
        print(f"acc = {acc}")
        return train_pred, train_feat, train_label

    def train(self, n_sample = 20):

        use_sam = False
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

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(training_steps * 0.06), num_training_steps = training_steps)

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
        criterion = nn.CrossEntropyLoss(reduction = 'mean')
        criterion = LabelSmoothingLoss(num_classes = self.num_labels, smoothing = 0.00, dim = -1, weight = None)
        # criterion = GCELoss(thresh=0.3, q = 0.7)

        # criterion = 
        # criterion = SCELoss(num_classes = self.num_labels, alpha = 0.1, beta = 1.0)
        best_model = None
        best_dev = -np.float('inf')
        best_test =  -np.float('inf')
        train_preds, train_losses, train_labels = [], [], []
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
                loss = outputs[0]
                logits = outputs[1]
                
                loss = criterion(pred = logits, target = batch[3].to(self.device))
                # print(loss, outputs[0])
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
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    epoch_iterator.set_description("iteration:%d, Loss:%.3f, best dev:%.3f" % (_, tr_loss/global_step, 100*best_dev))
                    if self.args.logging_steps > 0 and local_step in [training_len//2]: #   training_len//4,  training_len//4, training_len//2, 3*training_len//4  [100, 200, 300, 400, 500, 600]: # , ] and global_step % self.args.logging_steps == 0:
                        loss_dev, acc_dev, f1_dev = self.evaluate('dev', global_step)
                        # print(acc_dev)
                        loss_test, acc_test, f1_test = 0 ,0, 0
                        # loss_test, acc_test, f1_test = self.evaluate('test', global_step)
                        print("GLOBAL STEP", global_step, 'acc_dev:', acc_dev, 'f1', f1_dev, 'acc_test', acc_test, 'f1_test', f1_test)
                        if acc_dev > best_dev:
                            logger.info("Best model updated!")
                            self.best_model = copy.deepcopy(self.model.state_dict())
                            best_dev = acc_dev   
                        #### begin cartography #####
                        train_pred, train_feat, train_label = self.inference(train_only=True)
                        probs = train_pred[np.arange(len(train_label)), train_label]
                        loss = -np.log(probs)
                        train_preds.append(train_pred)
                        train_losses.append(loss)
                        #### end cartography #####             

                if 0 < training_steps < global_step:
                    epoch_iterator.close()
                    break
            #### begin cartography #####
            train_pred, train_feat, train_label = self.inference(train_only=True)
            probs = train_pred[np.arange(len(train_label)), train_label]
            loss = -np.log(probs)
            train_preds.append(train_pred)
            train_losses.append(loss)
            #### end cartography #####
            loss_dev, acc_dev, f1_dev = self.evaluate('dev', global_step)
            
            loss_test, acc_test, f1_test = 0 ,0, 0
            # loss_test, acc_test, f1_test = self.evaluate('test', global_step)
            if acc_dev > best_dev:
                logger.info("Best model updated!")
                self.best_model = copy.deepcopy(self.model.state_dict())
                best_dev = acc_dev
            print(f'Dev: Loss: {loss_dev}, Acc: {acc_dev}, F1: {f1_dev}', f'Test: Loss: {loss_test}, Acc: {acc_test}, F1: {f1_test}')
        #assert 0
        result_dict = {'seed': self.args.train_seed, 'labels': self.args.sample_labels}
        self.model.load_state_dict(self.best_model)
        loss_test, acc_test, acc_f1, preds_probs, out_label_ids = self.evaluate('test', global_step, return_preds = True)
        # {'acc': acc_test, 'lr': self.args.learning_rate, 'bsz': self.args.batch_size}
        result_dict['acc'] = acc_test
        result_dict['lr'] = self.args.learning_rate
        result_dict['bsz'] = self.args.batch_size
        if len(self.contra_datasets) > 0:
            for i, dataset in enumerate(self.contra_datasets):
                loss_, acc_i, f1_i = self.evaluate(mode = 'contra', dataset = dataset, global_step=global_step) 
                result_dict[f'acc_contra_{i}'] = acc_i
                result_dict[f'f1_contra_{i}'] = f1_i
                print(f'Test Contra {i}: Loss: {loss_}, Acc: {acc_i} , F1: {f1_i}')
        print(f'Test: Loss: {loss_test}, Acc: {acc_test}, F1: {acc_f1}')
        import json 
        line = json.dumps(result_dict)
        with open(f'{self.args.output_dir}_{self.args.model_type}_{self.args.semi_method}.json', 'a+') as f:
            f.write(line + '\n')
        print(np.array(train_losses).shape, np.array(train_preds).shape, train_label.shape)
        self.save_prediction(np.array(train_losses), np.array(train_preds), train_label, preds_probs, out_label_ids )
        self.save_model(stage = n_sample)
        return global_step, tr_loss / global_step
    
    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def get_monte_carlo_predictions(self, forward_passes, n_classes=2):
        """ Function to get the monte-carlo samples and uncertainty estimates
        through multiple forward passes

        Parameters
        ----------
        data_loader : object
            data loader object from the data loader module
        forward_passes : int
            number of monte-carlo samples/forward passes
        model : object
            keras model
        n_classes : int
            number of classes in the dataset
        n_samples : int
            number of samples in the test set
        """
        # n_classes = self.args.n_labels
        train_dataloader = DataLoader(self.train_dataset, shuffle=False, batch_size=self.args.eval_batch_size)

        unlabeled_dataloader = DataLoader(self.unlabeled, shuffle=False, batch_size=self.args.eval_batch_size)

        dropout_predictions = np.empty((0, len(self.unlabeled), n_classes))
        softmax = nn.Softmax(dim=1)
        for i in range(forward_passes):
            predictions = np.empty((0, n_classes))
            self.model.eval()
            self.enable_dropout()
            for batch in tqdm(unlabeled_dataloader, desc="Evaluating Labeled Set"):
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
                    preds = softmax(logits)
                predictions = np.vstack((predictions, preds.detach().cpu().numpy()))
            dropout_predictions = np.vstack((dropout_predictions,
                                            predictions[np.newaxis, :, :]))

        # for i in range(forward_passes):
        #     predictions = np.empty((0, n_classes))
        #     self.model.eval()
        #     self.enable_dropout()
        #     for i, (image, label) in enumerate(data_loader):

        #         image = image.to(torch.device('cuda'))
        #         with torch.no_grad():
        #             output = model(image)
        #             output = softmax(output) # shape (n_samples, n_classes)
        #         predictions = np.vstack((predictions, output.cpu().numpy()))

        #     dropout_predictions = np.vstack((dropout_predictions,
        #                                     predictions[np.newaxis, :, :]))

            # dropout predictions - shape (forward_passes, n_samples, n_classes)
        
        # Calculating mean across multiple MCD forward passes 
        mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes)

        # Calculating variance across multiple MCD forward passes 
        variance = np.var(dropout_predictions, axis=0) # shape (n_samples, n_classes)

        epsilon = 1e-13
        # Calculating entropy across multiple MCD forward passes 
        entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1) # shape (n_samples,)

        # Calculating mutual information across multiple MCD forward passes 
        mutual_info = entropy - np.mean(np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon),
                                                axis=-1), axis=0) # shape (n_samples,)
        print(mutual_info.shape)
        return mutual_info 

def get_mt_loss(s_logits, t_logits, class_name, _lambda):
    
    if class_name is None:
        return 0
    s_logits = s_logits.view(-1, s_logits.size(-1)).float()
    t_logits = t_logits.view(-1, t_logits.size(-1)).float()
    if class_name == "prob":
        logprob_stu = F.log_softmax(s_logits, 1)
        logprob_tea = F.log_softmax(t_logits, 1)
        return F.mse_loss(logprob_tea.exp(),logprob_stu.exp())*_lambda
    elif class_name == "logit":
        return F.mse_loss(s_logits.view(-1),t_logits.view(-1))*_lambda
    elif class_name == "smart":
        prob_stu = F.log_softmax(s_logits, 1).exp()
        prob_tea = F.log_softmax(t_logits, 1).exp()
        r_stu = -(1/(prob_stu+1e-6)-1+1e-6).detach().log()
        r_tea = -(1/(prob_tea+1e-6)-1+1e-6).detach().log()
        return (prob_stu*(r_stu-r_tea)*2).mean()*_lambda
    elif class_name == 'kl':
        logprob_stu = F.log_softmax(s_logits, 1)
        prob_tea = F.log_softmax(t_logits, 1).exp()
        return -(prob_tea*logprob_stu).sum(-1).mean()*_lambda
    elif class_name == 'distill':
        temp = 2
        logprob_stu = F.log_softmax(s_logits/temp, 1)
        prob_tea = F.log_softmax(t_logits/temp, 1).exp()
        return -(prob_tea*logprob_stu).sum(-1).mean()*_lambda


def mt_update(t_params, s_params, average="exponential", alpha=0.995, step=None):

    for (t_name, t_param), (s_name, s_param) in zip(t_params, s_params):
        if t_name != s_name:
            logger.error("t_name != s_name: {} {}".format(t_name, s_name))
            raise ValueError
        param_new = s_param.data.to(t_param.device)
        if average == "exponential":
            t_param.data.add_( (1-alpha)*(param_new-t_param.data) )
        elif average == "simple":
            virtual_decay = 1 / float(step)
            diff = (param_new - t_param.data) * virtual_decay
            t_param.data.add_(diff)

def opt_grad(loss, in_var, optimizer):
    
    if hasattr(optimizer, 'scalar'):
        loss = loss * optimizer.scaler.loss_scale
    return torch.autograd.grad(loss, in_var)

