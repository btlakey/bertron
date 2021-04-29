"""
authorship assignment papers:

http://publications.idiap.ch/downloads/papers/2020/Fabien_ICON2020_2020.pdf
https://link.springer.com/content/pdf/10.1007%2F978-3-540-30115-8_22.pdf
bert multiclass classsification:

https://towardsdatascience.com/multi-class-text-classification-with-deep-learning-using-bert-b59ca2f5c613
from: https://www.coursera.org/projects/sentiment-analysis-bert

"""

import json
import pandas as pd
import numpy as np
import random
from toolz import curry
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

SEED = 666
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def f1_score_func(preds, labels):
    """

    :param preds:
    :param labels:
    :return:
    """
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def accuracy_per_class(preds, labels):
    """

    :param preds:
    :param labels:
    :return:
    """
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])} / {len(y_true)}\n')


def input_dict(batch):
    """

    :param batch:
    :return:
    """
    return {
        'input_ids': batch[0],
        'attention_mask': batch[1],
        'labels': batch[2]
    }


def prepare_train(X):
    """

    :param X:
    :return:
    """

    # stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.index.values,
        X['label'].values,
        test_size=0.15,
        random_state=seed,
        stratify=X['label'].values
    )

    # train/test data labels
    X['data_type'] = np.nan * X.shape[0]
    X.loc[X_train, 'data_type'] = 'train'
    X.loc[X_test, 'data_type'] = 'test'

    # determine default sequence lengths
    vc = X['body'].apply(len).value_counts().sort_index()
    percents = [.9, .95, .98, .99]
    cs = map(lambda x: vc.cumsum() < vc.sum() * x, percents)

    lens = [vc[~i].index.min() for i in cs]
    points = dict(zip(percents, lens))

    # set length to include 95% of all messages (~2000 characters)
    max_len = points[0.95]
    # overwrite 512 default tensor setting
    max_len = 512

    return X, max_len


def make_tokenizer(X, max_len):
    """

    :param X:
    :param max_len:
    :return:
    """

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # partial evaluation (curry) for setting default values
    tokenizer_cur = curry(tokenizer.batch_encode_plus)(
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        max_length=max_len,
        truncation=True,
        return_tensors='pt'
    )

    # independent train and test toxenizers
    mask_train = X['data_type'] == 'train'
    encoded_data_train = tokenizer_cur(
        X[mask_train]['body'].values
    )
    mask_test = X['data_type'] == 'test'
    encoded_data_test = tokenizer_cur(
        X[mask_test]['body'].values
    )

    # index, attention masks and labels
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(X[mask_train]['label'].values)

    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(X[mask_test]['label'].values)

    # make train/test tensor datasets
    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

    return dataset_train, dataset_test


def evaluate(model, dataloader_test):
    """

    :param model:
    :param dataloader_test:
    :return:
    """
    model.eval()

    loss_test_total = 0
    predictions, true_tests = [], []

    for batch in dataloader_test:
        batch = tuple(b.to(DEVICE) for b in batch)
        inputs = input_dict(batch)

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_test_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_tests.append(label_ids)

    loss_test_avg = loss_test_total / len(dataloader_test)

    predictions = np.concatenate(predictions, axis=0)
    true_tests = np.concatenate(true_tests, axis=0)

    return loss_test_avg, predictions, true_tests


def train(model, dataset_train, dataset_test, label_dict, batch_size=16, epochs=2):

    # initiate model object
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_dict),
        output_attentions=False,
        output_hidden_states=False
    )

    # load train/test
    dataloader_train = DataLoader(
        dataset_train,
        sampler=RandomSampler(dataset_train),
        batch_size=batch_size
    )
    dataloader_test = DataLoader(
        dataset_test,
        sampler=SequentialSampler(dataset_test),
        batch_size=batch_size
    )

    # optimizer settings
    optimizer = AdamW(
        model.parameters(),
        lr=1e-5,
        eps=1e-8
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader_train) * epochs
    )

    # be there GPUs?
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    # train epochs
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()

        loss_train_total = 0
        progress_bar = tqdm(
            dataloader_train,
            desc='Epoch {:1d}'.format(epoch),
            leave=False,
            disable=False
        )

        for batch in progress_bar:
            model.zero_grad()

            batch = tuple(b.to(DEVICE) for b in batch)
            inputs = input_dict(batch)
            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

        torch.save(model.state_dict(), f'../models/finetuned_BERT_epoch_{epoch}.model')

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        test_loss, predictions, true_tests = evaluate(dataloader_test)
        test_f1 = f1_score_func(predictions, true_tests)
        tqdm.write(f'Test loss: {test_loss}')
        tqdm.write(f'F1 Score (Weighted): {test_f1}')

    return model


def main():

    X = X.read_parquet('../data/processed/X.parquet')
    with open('../data/processed/labels.json', 'w') as f_in:
        label_dict = json.load(f_in)

    X, max_len = prepare_train(X)
    dataset_train, dataset_test = make_tokenizer(X, max_len)

    batch_size=16
    epochs=2
    model = train(
        dataset_train,
        dataset_test,
        label_dict,
        batch_size=batch_size,
        epochs=epochs
    )

    _, predictions, true_tests = evaluate(model, dataloader_test)
    accuracy_per_class(predictions, true_tests)


if __name__ == '__main__':
    main()
