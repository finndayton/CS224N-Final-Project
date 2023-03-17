import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
from pcgrad import PCGrad

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask, sentiment_eval, paraphrase_eval, similarity_eval, nli_eval


TQDM_DISABLE=False

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        self.sentiment_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_ln = nn.Linear(config.hidden_size, 5)

        self.paraphrase_ln = nn.Linear(config.hidden_size * 2, 1)
        self.paraphrase_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.similarity_ln = nn.Linear(config.hidden_size * 2, 1)
        self.similarity_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.nli_ln = nn.Linear(config.hidden_size * 2, 3)
        self.nli_dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        bert_output = self.bert(input_ids, attention_mask)
        return bert_output["pooler_output"]


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        bert_output = self.forward(input_ids, attention_mask)
        outputs = self.sentiment_dropout(bert_output)
        probs = self.sentiment_ln(outputs)
        return probs


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO           
        bert_output_1 = self.forward(input_ids_1, attention_mask_1)
        bert_output_2 = self.forward(input_ids_2, attention_mask_2)

        # Concatenate the pooled outputs
        concatenated_output = torch.cat([bert_output_1, bert_output_2], dim=-1)

        # Apply dropout
        outputs = self.paraphrase_dropout(concatenated_output)

        return self.paraphrase_ln(outputs).squeeze()

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        bert_output_1 = self.forward(input_ids_1, attention_mask_1)
        bert_output_2 = self.forward(input_ids_2, attention_mask_2)

        # Concatenate the pooled outputs
        concatenated_output = torch.cat([bert_output_1, bert_output_2], dim=-1)
        
        # Apply dropout
        outputs = self.similarity_dropout(concatenated_output)

        return self.similarity_ln(outputs).squeeze()

    def predict_nli(self,
                        input_ids_1, attention_mask_1,
                        input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs 3 logits for each of
        the 3 classes: 'neutral', 'entailment', 'contradiction'
        '''
        ### TODO
        bert_output_1 = self.forward(input_ids_1, attention_mask_1)
        bert_output_2 = self.forward(input_ids_2, attention_mask_2)

        # Concatenate the pooled outputs
        concatenated_output = torch.cat([bert_output_1, bert_output_2], dim=-1)
        outputs = self.nli_dropout(concatenated_output)

        return self.nli_ln(outputs)


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def pretrain_nli(args):
    save_path = args.nli_filepath

    batch_size_nli = 16

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_sentiment_labels,para_train_data, sts_train_data, multinli_train_data, num_multinli_labels = load_multitask_data(args.sst_train,args.para_train,args.sts_train, args.multinli_train, split ='train', nli_limit=args.nli_limit)
    sst_dev_data, num_sentiment_labels,para_dev_data, sts_dev_data, multinli_dev_data, num_multinli_labels = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, args.multinli_dev, split ='train', nli_limit=args.nli_limit)

    multinli_train_data = SentencePairDataset(multinli_train_data, args)
    multinli_dev_data = SentencePairDataset(multinli_dev_data, args)

    multinli_train_dataloader = DataLoader(multinli_train_data, shuffle=True, batch_size=batch_size_nli,
                                      collate_fn=multinli_train_data.collate_fn)
    multinli_dev_dataloader = DataLoader(multinli_dev_data, shuffle=False, batch_size=batch_size_nli,
                                    collate_fn=multinli_dev_data.collate_fn)


    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_sentiment_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': 'finetune'}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(multinli_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'], 
                                       batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_nli(b_ids_1, b_mask_1, b_ids_2, b_mask_2)

            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / batch_size_nli

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = nli_eval(multinli_train_dataloader, model, device)
        dev_acc, dev_f1, *_ = nli_eval(multinli_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, save_path)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_sentiment_labels,para_train_data, sts_train_data, multinli_train_data, num_multinli_labels = load_multitask_data(args.sst_train,args.para_train,args.sts_train, args.multinli_train, split ='train')
    sst_dev_data, num_sentiment_labels,para_dev_data, sts_dev_data, multinli_dev_data, num_multinli_labels = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, args.multinli_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    #sst
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    

    if args.nli:
        # Get model from pretrain
        saved = torch.load(args.nli_filepath)
        config = saved['model_config']
        config['finetune'] = 'finetune'
        model = MultitaskBERT(config)
    else:
        # Init model
        config = {'hidden_dropout_prob': args.hidden_dropout_prob,
                'num_labels': num_sentiment_labels,
                'hidden_size': 768,
                'data_dir': '.',
                'option': 'finetune'}

        config = SimpleNamespace(**config)

        model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def train_multitask_gradient_surgery(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    batch_size_sst = 16 if args.gs_wrap else 1
    batch_size_para = 16 if args.gs_wrap else 24
    batch_size_sts = 16 if args.gs_wrap else 1

    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_sentiment_labels,para_train_data, sts_train_data, multinli_train_data, num_multinli_labels = load_multitask_data(args.sst_train,args.para_train,args.sts_train, args.multinli_train, split ='train')
    sst_dev_data, num_sentiment_labels,para_dev_data, sts_dev_data, multinli_dev_data, num_multinli_labels = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, args.multinli_dev, split ='train')

    sst_dataset_len = len(sst_train_data)
    sts_dataset_len = len(sts_train_data)
    para_dataset_len = len(para_train_data)

    n = max(sst_dataset_len, sts_dataset_len, para_dataset_len)
    max_len = n if args.gs_wrap else None
    
    # Build DataLoaders
    # sst
    sst_train_data = SentenceClassificationDataset(sst_train_data, args, max_len=max_len)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=batch_size_sst,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=batch_size_sst,
                                    collate_fn=sst_dev_data.collate_fn)

    # sts
    sts_train_data = SentencePairDataset(sts_train_data, args, max_len=max_len)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=batch_size_sts,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=batch_size_sts,
                                    collate_fn=sts_dev_data.collate_fn)

    # quora
    para_train_data = SentencePairDataset(para_train_data, args, max_len=max_len)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=batch_size_para,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=batch_size_para,
                                    collate_fn=para_dev_data.collate_fn)


    sst_dataloader_len = len(sst_train_dataloader)
    sts_dataloader_len = len(sts_train_dataloader)
    para_dataloader_len = len(para_train_dataloader)

    min_dataloader_len = min(sst_dataloader_len, sts_dataloader_len, para_dataloader_len)

    print(f"\n sst_train_dataloader: {sst_dataloader_len}, sts_train_dataloader: {sts_dataloader_len}, para_train_dataloader: {para_dataloader_len}\n")

    
    if args.nli:
        # Load model from nli
        saved = torch.load(args.nli_filepath)
        config = saved['model_config']
        config['option'] = 'finetune'
        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])

        if args.gs_wrap:
            save_path = args.nli_gs_wrap_filepath
        else:
            save_path = args.nli_gs_batch_diff_filepath
    
    else:
        # Init model
        config = {'hidden_dropout_prob': args.hidden_dropout_prob,
                'num_labels': num_sentiment_labels,
                'hidden_size': 768,
                'data_dir': '.',
                'option': 'finetune'}

        config = SimpleNamespace(**config)

        model = MultitaskBERT(config)

        if args.gs_wrap:
            save_path = args.gs_wrap_filepath
        else:
            save_path = args.gs_batch_diff_filepath

    model = model.to(device)

    lr = args.lr
    pc_adam = PCGrad(AdamW(model.parameters(), lr=lr))
    best_dev_sst_acc = 0
    best_dev_para_acc = 0
    best_dev_sts_corr = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss_sst, train_loss_sts, train_loss_para = 0,0,0
        num_batches = 0

        def gradient_surgery_batch_step(batch_sst, batch_sts, batch_para):
            # Calculate loss for SST
            b_ids_sst, b_mask_sst, b_labels_sst = (batch_sst['token_ids'],
                                       batch_sst['attention_mask'], batch_sst['labels'])
            b_ids_sst, b_mask_sst, b_labels_sst = b_ids_sst.to(device),  b_mask_sst.to(device), b_labels_sst.to(device)
            logits = model.predict_sentiment(b_ids_sst, b_mask_sst)
            loss_sst = F.cross_entropy(logits, b_labels_sst.view(-1), reduction='sum') / batch_size_sst

            # Calculate loss for STS
            b_ids_sts_1, b_mask_sts_1, b_ids_sts_2, b_mask_sts_2, b_labels_sts = (batch_sts['token_ids_1'], batch_sts['attention_mask_1'], 
                                       batch_sts['token_ids_2'], batch_sts['attention_mask_2'], batch_sts['labels'])
            b_ids_sts_1, b_mask_sts_1, b_ids_sts_2, b_mask_sts_2, b_labels_sts = b_ids_sts_1.to(device), b_mask_sts_1.to(device), b_ids_sts_2.to(device), b_mask_sts_2.to(device), b_labels_sts.to(device)
            logits = model.predict_similarity(b_ids_sts_1, b_mask_sts_1, b_ids_sts_2, b_mask_sts_2)
            loss_sts = nn.MSELoss()(logits, b_labels_sts.float()) / batch_size_sts
        
            # Calculate loss for PARA
            b_ids_para_1, b_mask_para_1, b_ids_para_2, b_mask_para_2, b_labels_para = (batch_para['token_ids_1'], batch_para['attention_mask_1'], 
                                       batch_para['token_ids_2'], batch_para['attention_mask_2'], batch_para['labels'])
            b_ids_para_1, b_mask_para_1, b_ids_para_2, b_mask_para_2, b_labels_para = b_ids_para_1.to(device), b_mask_para_1.to(device), b_ids_para_2.to(device), b_mask_para_2.to(device), b_labels_para.to(device)
            logits = model.predict_paraphrase(b_ids_para_1, b_mask_para_1, b_ids_para_2, b_mask_para_2)
            b_labels_para = b_labels_para.float()
            loss_para_fn = nn.BCEWithLogitsLoss()
            loss_para = nn.BCEWithLogitsLoss()(logits, b_labels_para.float()) / batch_size_para

            pc_adam.pc_backward([loss_sst, loss_sts, loss_para/batch_size_para])
            pc_adam.step()

            return loss_sst, loss_sts, loss_para

        # GS Wrap around
        if args.gs_wrap:
            print(f"Performing GS_WRAP. Double check dataloader lengths: sst_train_dataloader: {len(sst_train_dataloader)}, sts_train_dataloader: {len(sts_train_dataloader)}, para_train_dataloader: {len(para_train_dataloader)}")
            for batch_sst, batch_sts, batch_para in tqdm(zip(sst_train_dataloader, sts_train_dataloader, para_train_dataloader), desc=f'train-{epoch}', disable=TQDM_DISABLE):
                
                loss_sst, loss_sts, loss_para = gradient_surgery_batch_step(batch_sst, batch_sts, batch_para)

                train_loss_sst += loss_sst.item()
                train_loss_sts += loss_sts.item()
                train_loss_para += loss_para.item()

                num_batches += 1

        # GS Different Batch Sizes
        else: 
            print(f"Performing GS_BATCH_DIFF. Double check dataloader lengths: sst_train_dataloader: {len(sst_train_dataloader)}, sts_train_dataloader: {len(sts_train_dataloader)}, para_train_dataloader: {len(para_train_dataloader)}")
            sst_train_dataloader_iter = iter(sst_train_dataloader)
            sts_train_dataloader_iter = iter(sts_train_dataloader)
            para_train_dataloader_iter = iter(para_train_dataloader)
            for i in tqdm(range(min_dataloader_len), desc=f'train-{epoch}', disable=TQDM_DISABLE):
                batch_sst = next(sst_train_dataloader_iter)
                batch_sts = next(sts_train_dataloader_iter)
                batch_para = next(para_train_dataloader_iter)

                loss_sst, loss_sts, loss_para = gradient_surgery_batch_step(batch_sst, batch_sts, batch_para)

                train_loss_sst += loss_sst.item()
                train_loss_sts += loss_sts.item()
                train_loss_para += loss_para.item()

                num_batches += 1

        train_loss_sst = train_loss_sst / (num_batches)
        train_loss_sts = train_loss_sts / (num_batches)
        train_loss_para = train_loss_para / (num_batches)

        print(f"\ntrain eval:\n")
        train_eval = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        print(f"\ndev eval:\n")
        dev_eval = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        train_sst_acc, train_para_acc, train_sts_corr = train_eval[3], train_eval[0], train_eval[6]    
        dev_sst_acc, dev_para_acc, dev_sts_corr = dev_eval[3], dev_eval[0], dev_eval[6]

        if dev_sst_acc >= best_dev_sst_acc and dev_para_acc >= best_dev_para_acc and dev_sts_corr >= best_dev_sts_corr:
            best_dev_sst_acc = dev_sst_acc
            best_dev_para_acc = dev_para_acc
            best_dev_sts_corr = dev_sts_corr
            save_model(model, pc_adam._optim, args, config, save_path)

        print(f"Epoch {epoch}: train loss SST :: {train_loss_sst :.3f}, train acc :: {train_sst_acc :.3f}, dev acc :: {dev_sst_acc :.3f}")
        print(f"Epoch {epoch}: train loss PARA :: {train_loss_para :.3f}, train acc :: {train_para_acc :.3f}, dev acc :: {dev_para_acc :.3f}")
        print(f"Epoch {epoch}: train loss STS :: {train_loss_sts :.3f}, train acc :: {train_sts_corr :.3f}, dev acc :: {dev_sts_corr :.3f}")
        

def train_final_layers(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # Load data
    sst_train_data, num_sentiment_labels,para_train_data, sts_train_data, multinli_train_data, num_multinli_labels = load_multitask_data(args.sst_train,args.para_train,args.sts_train, args.multinli_train, split ='train')
    sst_dev_data, num_sentiment_labels,para_dev_data, sts_dev_data, multinli_dev_data, num_multinli_labels = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, args.multinli_dev, split ='train')

    sst_dataset_len = len(sst_train_data)
    sts_dataset_len = len(sts_train_data)
    para_dataset_len = len(para_train_data)

    print(f"\nsst_train_data: {sst_dataset_len}, sts_train_data: {sts_dataset_len}, para_train_data: {para_dataset_len}\n")
    
    # sst
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    # sts
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    # quora
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    print(f"\nsst_train_data: {len(sst_train_data)}, sts_train_data: {len(sts_train_data)}, para_train_data: {len(para_train_data)}\n")

    #sst
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    #sts
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)
    #quora 
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)

    
    if args.nli and args.gs_wrap:
        save_path = args.nli_gs_wrap_final_layer_filepath
        saved = torch.load(args.nli_gs_wrap_filepath)
    elif args.nli and args.gs_batch_diff:
        save_path = args.nli_gs_batch_diff_final_layer_filepath
        saved = torch.load(args.nli_gs_batch_diff_filepath)
    elif args.gs_wrap:
        save_path = args.gs_wrap_final_layer_filepath
        saved = torch.load(args.gs_wrap_filepath)
    elif args.gs_batch_diff:
        save_path = args.gs_batch_diff_final_layer_filepath
        saved = torch.load(args.gs_batch_diff_filepath)
    elif args.nli:
        save_path = args.nli_final_layer_filepath
        saved = torch.load(args.nli_filepath)
    else:
        save_path = args.final_layer_filepath
        saved = None

    if saved:
        config = saved['model_config']
        config['option'] = 'pretrain'
        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
    else:
        # Init model
        config = {'hidden_dropout_prob': args.hidden_dropout_prob,
                'num_labels': num_sentiment_labels,
                'hidden_size': 768,
                'data_dir': '.',
                'option': 'pretrain'}
                
        config = SimpleNamespace(**config)
        model = MultitaskBERT(config)

    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    # best_dev_acc = 0

    def finetune_layer(name, train_dataloader, dev_dataloader, eval_fn, save_path):
        # Run for the specified number of epochs
        best_dev_acc = 0
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            num_batches = 0
            for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                optimizer.zero_grad()
                # SST 
                if (name == "sst"):
                    b_ids, b_mask, b_labels = (batch['token_ids'],
                                            batch['attention_mask'], batch['labels'])
                    b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
                    logits = model.predict_sentiment(b_ids, b_mask)
                    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                # Quora and STS
                else:
                    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'], 
                                       batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
                    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = b_ids_1.to(device), b_mask_1.to(device), b_ids_2.to(device), b_mask_2.to(device), b_labels.to(device)
                    if name == "sts":
                        logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                        loss = nn.MSELoss()(logits, b_labels.float()) / args.batch_size
                    elif name == "quora":
                        logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                        loss = nn.BCEWithLogitsLoss()(logits, b_labels.float()) / args.batch_size
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss = train_loss / (num_batches)

            train_acc, train_f1, *_ = eval_fn(train_dataloader, model, device)
            dev_acc, dev_f1, *_ = eval_fn(dev_dataloader, model, device)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_model(model, optimizer, args, config, save_path)

            print(f"Epoch {epoch} for {name}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

    print("Finetuning SST Layer")
    finetune_layer("sst", sst_train_dataloader, sst_dev_dataloader, sentiment_eval, save_path)
    print("Finetuning QUORA Layer")
    finetune_layer("quora", para_train_dataloader, para_dev_dataloader, paraphrase_eval, save_path)
    print("Finetuning STS Layer")
    finetune_layer("sts", sts_train_dataloader, sts_dev_dataloader, similarity_eval, save_path)

def test_model(args):
    if args.nli and args.gs_wrap and args.final_layer:
        load_filepath = args.nli_gs_wrap_final_layer_filepath
    elif args.nli and args.gs_batch_diff and args.final_layer:
        load_filepath = args.nli_gs_batch_diff_final_layer_filepath
    elif args.gs_wrap and args.final_layer:
        load_filepath = args.gs_wrap_final_layer_filepath
    elif args.gs_batch_diff and args.final_layer:
        load_filepath = args.gs_batch_diff_final_layer_filepath
    elif args.nli and args.final_layer:
        load_filepath = args.nli_final_layer_filepath
    elif args.nli and args.gs_wrap:
        load_filepath = args.nli_gs_wrap_filepath
    elif args.nli and args.gs_batch_diff:
        load_filepath = args.nli_gs_batch_diff_filepath
    elif args.final_layer:
        load_filepath = args.final_layer_filepath
    elif args.gs_wrap:
        load_filepath = args.gs_wrap_filepath
    elif args.gs_batch_diff:
        load_filepath = args.gs_batch_diff_filepath
    elif args.nli:
        load_filepath = args.nli_filepath


    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        if args.test_model:
            saved = torch.load(args.test_model)
        else:
            saved = torch.load(load_filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--multinli_train", type=str, default="data/multinli_1.0/multinli_1.0_train.jsonl")
    parser.add_argument("--multinli_dev", type=str, default="data/multinli_1.0/multinli_1.0_dev_matched.jsonl")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    

    parser.add_argument("--test_model", help='', type=str, default=None)

    # For the below flags, pass in 'train' or 'load'
    parser.add_argument("--nli", help='', type=str, default=None)
    parser.add_argument("--gs_wrap", help='', type=str, default=None)
    parser.add_argument("--gs_batch_diff", help='', type=str, default=None)
    parser.add_argument("--final_layer", help='', type=str, default=None)

    # In case we want to train nli on less examples
    parser.add_argument("--nli_limit", help='', type=int, default=None)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    # Possible Model Filepaths
    args.nli_filepath = f'nli-{args.epochs}-{args.lr}.pt'
    args.gs_batch_diff_filepath = f'gs_batch_diff-{args.epochs}-{args.lr}.pt'
    args.gs_wrap_filepath = f'gs_wrap-{args.epochs}-{args.lr}.pt'
    args.final_layer_filepath = f'final_layer-{args.epochs}-{args.lr}.pt'

    args.nli_gs_batch_diff_filepath = f'nli-gs_batch_diff-{args.epochs}-{args.lr}.pt'
    args.nli_gs_wrap_filepath = f'nli-gs_wrap-{args.epochs}-{args.lr}.pt'
    args.nli_final_layer_filepath = f'nli-final_layer-{args.epochs}-{args.lr}.pt'

    args.gs_batch_diff_final_layer_filepath = f'gs_batch_diff-final_layer-{args.epochs}-{args.lr}.pt'
    args.gs_wrap_final_layer_filepath = f'gs_wrap-final_layer-{args.epochs}-{args.lr}.pt'

    args.nli_gs_batch_diff_final_layer_filepath = f'nli-gs_batch_diff-final_layer-{args.epochs}-{args.lr}.pt'
    args.nli_gs_wrap_final_layer_filepath = f'nli-gs_wrap-final_layer-{args.epochs}-{args.lr}.pt'


    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path 
    seed_everything(args.seed)  # fix the seed for reproducibility
    if args.test_model is None:
        if args.nli == 'train':
            pretrain_nli(args)

        if args.gs_batch_diff == 'train' or args.gs_wrap == 'train':
            train_multitask_gradient_surgery(args)

        if args.final_layer == 'train':
            train_final_layers(args)

    print(f"\ntesting model commencing\n")
    test_model(args)
