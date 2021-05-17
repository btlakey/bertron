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
from tqdm import tqdm
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
)

SEED = 666
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# be there GPUs?
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def f1_score_func(preds, labels):
    """calculate f1 score given nested numpy arrays

    :param preds: numpy.ndarray
        predictions, dim=()
    :param labels: numpy.ndarray
        truth values, dim()

    :return: float
        f1 score
    """
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def accuracy_per_class(preds, labels, label_dict):
    """ class (author) accuracy (proportion true predictions: tp+tn / tp+tn+fp+fn)

    :param preds: numpy.ndarray
        predictions, dim=()
    :param labels: numpy.ndarray
        truth values, dim()
    :param label_dict: dict
        mapping between authors (key) and int labels (value)

    :return: None
        print class accuracy metrics
    """
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        label_mask = labels_flat == label
        y_preds = preds_flat[label_mask]
        y_true = labels_flat[label_mask]
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


def sequence_len(srs, max_len_prop=0.95):
    """ Return max_seq_len for training, with the aim of truncating only (1-max_len_prop)
    Currently set to max_len = 512 as a result of pretrained BERT model

    :param srs: pandas.Series
        series of input text
    :param max_len_prop: float
        between 0-1, what proportion of records should not be trimmed

    :return: int
        max_len for text sequences
    """

    # determine default sequence lengths
    vc = srs.apply(len).value_counts().sort_index()

    # find how many characters are required to account for each proportion of recrods
    percents = [.9, .95, .98, .99]
    cs = map(lambda x: vc.cumsum() < vc.sum() * x, percents)

    lens = [vc[~i].index.min() for i in cs]
    points = dict(zip(percents, lens))

    # set length to include 95% of all messages (~2000 characters)
    max_len = int(points[max_len_prop])

    # TODO: overwrite 512 default tensor setting, ths comes from pretrained BERT model
    max_len = 512

    return max_len


def prepare_train(X):
    """ Trim input text to max_len and designate train/test

    :param X: pandas.DataFrame
        dataframe, read from disk as output of format_data.py

    :return: pandas.DataFrame
        input dataframe, with additional 'data_type' column specifying train/test (stratified
        on label='author') and 'body' text field trimmed to max_len
    """

    # stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.index.values,
        X['label'].values,
        test_size=0.15,
        random_state=SEED,
        stratify=X['label'].values
    )

    # train/test data labels
    X['data_type'] = np.nan * X.shape[0]
    X.loc[X_train, 'data_type'] = 'train'
    X.loc[X_test, 'data_type'] = 'test'

    max_len = sequence_len(X['body'])

    return X, max_len


def tokenize_data(X, max_len):
    """ Tokenize input data and return tokenized tensor datasets
    Uses pretrained 'bert-base-uncased' from transformers library

    :param X: pandas.DataFrame
        input dataframe with 'body' field designating input text and 'data_type' field
        designating train/test records
    :param max_len: int
        max_len for input sequences

    :return: (
        torch.utils.data.dataset.TensorDataset,
        torch.utils.data.dataset.TensorDataset
    )
        train and test torch tensor datasets
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
    """ Given a pass over an epoch, report out-of-sample test metrics

    :param model: transformers.models.bert.modeling_bert.BertForSequenceClassification
        fine-tuned BERT sequence classifier
    :param dataloader_test: torch.utils.data.dataloader.DataLoader
        iterable over test TensorDataset

    :return: (float, numpy.ndarray, numpy.ndarray)
        loss_test_avg: average loss over test records
        predictions: array of predicted labels
        true_tests: array or truth labels
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


def train(dataset_train, dataset_test, label_dict, model_params):
    """ Fine-tune a BERT sequence classifier for author attribution using the Enron corpus

    :param model: transformers.models.bert.modeling_bert.BertForSequenceClassification
        pretrained model from transformers library for finetunig
    :param dataset_train: torch.utils.data.dataset.TensorDataset
        map-style train tensor dataset, output of tokenize_data()
    :param dataset_test: torch.utils.data.dataset.TensorDataset
        map-style test tensor dataset, output of tokenize_data()
    :param label_dict: dict
        dictionary mapping numeric labels to authors
    :param batch_size: int
    :param epochs: int

    :return: transformers.models.bert.modeling_bert.BertForSequenceClassification
        fine-tuned BERT sequence classifier
    """

    try:
        batch_size = model_params['batch_size']
        epochs = model_params['epochs']
        lr = model_params['learning_rate']
        eps = model_params['epsilon']
    except KeyError as e:
        raise Exception("Must include batch_size, epochs, learning_rate, and epsilon values"
                        "when calling train()")

    # initiate model object
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_dict),
        output_attentions=False,
        output_hidden_states=False,
        # max_position_embeddings=max_len
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

    model.to(DEVICE)

    # train epochs
    print('\nis tqdm going to work to stdout?')
    for epoch in tqdm(range(1, epochs + 1),  file=sys.stdout):
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

        # periodic saves
        torch.save(model.state_dict(), f'models/finetuned_BERT_epoch_{epoch}.model')

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        test_loss, predictions, true_tests = evaluate(model, dataloader_test)
        test_f1 = f1_score_func(predictions, true_tests)
        tqdm.write(f'Test loss: {test_loss}')
        tqdm.write(f'F1 Score (Weighted): {test_f1}')

    return model


#TODO: make these click args
def main(batch_size=32, epochs=4):
    """ main invocation

    :param batch_size: int
    :param epochs: int
    :return: None
        saved model to disk
    """

    print('\npreparing dataset')
    X = pd.read_parquet('../data/processed/X.parquet')
    with open('../data/processed/labels.json', 'r') as f_in:
        label_dict = json.load(f_in)

    print('\npreparing datasets')
    X, max_len = prepare_train(X)
    dataset_train, dataset_test = tokenize_data(X, max_len)

    print('\npreparing to train BERT model')
    model_params = {
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': 1e-4,
        'epsilon': 1e-8
    }
    model = train(
        dataset_train,
        dataset_test,
        label_dict,
        model_params
    )

    torch.save(model.state_dict(), f'models/finetuned_BERT_final.model')


# invocation: python train_bert.py > model_output.txt
if __name__ == '__main__':
    main()
