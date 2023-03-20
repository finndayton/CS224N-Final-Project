#!/usr/bin/env python3

'''
Model evaluation functions.

When training your multitask model, you will find it useful to run
model_eval_multitask to be able to evaluate your model on the 3 tasks in the
development set.

Before submission, your code needs to call test_model_multitask(args, model, device) to generate
your predictions. We'll evaluate these predictions against our labels on our end,
which is how the leaderboard will be updated.
The provided test_model() function in multitask_classifier.py **already does this for you**,
so unless you change it you shouldn't need to call anything from here
explicitly aside from model_eval_multitask.
'''

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_multitask_data, load_multitask_test_data, \
    SentenceClassificationDataset, SentenceClassificationTestDataset, \
    SentencePairDataset, SentencePairTestDataset


TQDM_DISABLE = False

# Evaluate a multitask model for accuracy.on SST only.
def model_eval_sst(dataloader, model, device):
    model.eval()  # switch to eval model, will turn off randomness like dropout
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'],batch['attention_mask'],  \
                                                        batch['labels'], batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model.predict_sentiment(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents, sent_ids

def sentiment_eval(sentiment_dataloader, model, device, doAnalysis=False, train_dataset=None):
    model.eval()  # switch to eval model, will turn off randomness like dropout
    analysis = []
    with torch.no_grad():
        sst_y_true = []
        sst_y_pred = []
        sst_sent_ids = []

        # Evaluate sentiment classification.
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_labels, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['labels'], batch['sent_ids']
            b_sents = batch['sents']
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            logits = model.predict_sentiment(b_ids, b_mask)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            # Record example for analysis
            if doAnalysis:
                for i in range(len(b_ids)):
                    example = {}
                    example['len'] = len(b_sents[i])
                    example['label'] = b_labels[i]
                    example['pred'] = y_hat[i]
                    example['rank'] = sum([ 0 if id.item() not in train_dataset.rank_map else train_dataset.rank_map[id.item()] for id in b_ids[i]])
                    example['accurate'] = 1 if b_labels[i] == y_hat[i] else 0
                    analysis.append(example)

            sst_y_pred.extend(y_hat)
            sst_y_true.extend(b_labels)
            sst_sent_ids.extend(b_sent_ids)

        if doAnalysis:
            analysis_df = pd.DataFrame(analysis)
            # Plot length by accuracy
            len_df = analysis_df.groupby('len', as_index=False).agg({'accurate': 'mean'})
            n_bins = 50
            hist, bins, _ = plt.hist(len_df['len'], bins=n_bins)
            len_df['bin'] = pd.cut(len_df['len'], bins=bins, labels=bins[:-1])
            len_df = len_df.groupby('bin', as_index=False).agg({'accurate': 'mean'})
            len_df['bin'] = len_df['bin'].astype(int)
            len_df.plot(kind='line', x='bin', y='accurate')
            plt.title(f'Sentiment Accuracy by Sentence Length')
            plt.xlabel('Sentence Length')
            plt.ylabel('Accuracy')
            plt.savefig(f'sentiment_predictions_by_length.png')

            # Plot rank by accuracy
            rank_df = analysis_df.groupby('rank', as_index=False).agg({'accurate': 'mean'})
            n_bins = 20
            hist, bins, _ = plt.hist(rank_df['rank'], bins=n_bins)
            rank_df['bin'] = pd.cut(rank_df['rank'], bins=bins, labels=bins[:-1])
            rank_df = rank_df.groupby('bin', as_index=False).agg({'accurate': 'mean'})
            rank_df['bin'] = rank_df['bin'].astype(int)
            rank_df.plot(kind='line', x='bin', y='accurate')
            plt.title(f'Sentiment Accuracy by Sentence Rarity')
            plt.xlabel('Rarity Rank (Sentences with common words have higher rank)')
            plt.ylabel('Accuracy')
            plt.savefig(f'sentiment_predictions_by_rank.png')

            # Plot label by accuracy
            groupByLabel = analysis_df.groupby('label', as_index=False).agg({'accurate': 'mean'})
            groupByLabel['label'] = groupByLabel['label'].astype('category')
            groupByLabel.plot(kind='bar', x='label', y='accurate')
            plt.xlabel('Label')
            plt.ylabel('Accuracy')
            plt.xticks(rotation='horizontal')
            plt.savefig(f'sentiment_predictions_by_label.png')

        sentiment_accuracy = np.mean(np.array(sst_y_pred) == np.array(sst_y_true))
        return sentiment_accuracy,sst_y_pred, sst_sent_ids

def paraphrase_eval(paraphrase_dataloader, model, device, train_dataset=None):
    model.eval()  # switch to eval model, will turn off randomness like dropout
    with torch.no_grad():
        para_y_true = []
        para_y_pred = []
        para_sent_ids = []

        # Evaluate paraphrase detection.
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_y_true.extend(b_labels)
            para_sent_ids.extend(b_sent_ids)

        paraphrase_accuracy = np.mean(np.array(para_y_pred) == np.array(para_y_true))
        return paraphrase_accuracy, para_y_pred, para_sent_ids

def similarity_eval(sts_dataloader, model, device, train_dataset=None):
    model.eval()  # switch to eval model, will turn off randomness like dropout
    with torch.no_grad():
        sts_y_true = []
        sts_y_pred = []
        sts_sent_ids = []


        # Evaluate semantic textual similarity.
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_y_true.extend(b_labels)
            sts_sent_ids.extend(b_sent_ids)
        pearson_mat = np.corrcoef(sts_y_pred,sts_y_true)
        sts_corr = pearson_mat[1][0]
        return sts_corr, sts_y_pred, sts_sent_ids

def nli_eval(nli_dataloader, model, device):
    model.eval()  # switch to eval model, will turn off randomness like dropout
    with torch.no_grad():
        nli_y_true = []
        nli_y_pred = []
        nli_sent_ids = []

        # Evaluate nli detection.
        for step, batch in enumerate(tqdm(nli_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_nli(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            nli_y_pred.extend(y_hat)
            nli_y_true.extend(b_labels)
            nli_sent_ids.extend(b_sent_ids)

        nli_accuracy = np.mean(np.array(nli_y_pred) == np.array(nli_y_true))
        return nli_accuracy, nli_y_pred, nli_sent_ids


# Perform model evaluation in terms by averaging accuracies across tasks.
def model_eval_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device, 
                         train_sentiment_dataset=None, train_paraphrase_dataset=None, train_sts_dataset=None):
        sentiment_accuracy,sst_y_pred, sst_sent_ids = sentiment_eval(sentiment_dataloader, model, device, doAnalysis=True, train_dataset=train_sentiment_dataset)
        paraphrase_accuracy, para_y_pred, para_sent_ids = paraphrase_eval(paraphrase_dataloader, model, device, train_dataset=train_paraphrase_dataset)
        sts_corr, sts_y_pred, sts_sent_ids = similarity_eval(sts_dataloader, model, device, train_dataset=train_sts_dataset)

        print(f'Paraphrase detection accuracy: {paraphrase_accuracy:.3f}')
        print(f'Sentiment classification accuracy: {sentiment_accuracy:.3f}')
        print(f'Semantic Textual Similarity correlation: {sts_corr:.3f}')

        return (paraphrase_accuracy, para_y_pred, para_sent_ids,
                sentiment_accuracy,sst_y_pred, sst_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids)

# Perform model evaluation in terms by averaging accuracies across tasks.
def model_eval_test_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device, focus=None):
    model.eval()  # switch to eval model, will turn off randomness like dropout

    with torch.no_grad():

        para_y_pred = []
        para_sent_ids = []
        # Evaluate paraphrase detection.
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_sent_ids.extend(b_sent_ids)


        sts_y_pred = []
        sts_sent_ids = []


        # Evaluate semantic textual similarity.
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_sent_ids.extend(b_sent_ids)


        sst_y_pred = []
        sst_sent_ids = []

        # Evaluate sentiment classification.
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'],  batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            logits = model.predict_sentiment(b_ids, b_mask)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_sent_ids.extend(b_sent_ids)

        return (para_y_pred, para_sent_ids,
                sst_y_pred, sst_sent_ids,
                sts_y_pred, sts_sent_ids)


def test_model_multitask(args, model, device):
        
        sst_train_data, num_sentiment_labels,para_train_data, sts_train_data, multinli_train_data, num_multinli_labels = load_multitask_data(args.sst_train,args.para_train,args.sts_train, args.multinli_train, split ='train', nli_limit=args.nli_limit)

        sst_test_data, num_labels,para_test_data, sts_test_data, nli_data, nli_labels = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, args.multinli_train, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data, nli_dev_data, nli_dev_labels = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, args.multinli_dev, split='dev')

        sst_train_data = SentenceClassificationDataset(sst_train_data, args, rank_map=True)
        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args) 

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, dev_sts_corr, \
            dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device,
                                                                    train_sentiment_dataset=sst_train_data)

        test_para_y_pred, test_para_sent_ids, test_sst_y_pred, \
            test_sst_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")
