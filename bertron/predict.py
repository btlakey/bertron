"""
TODO: not quite working
"""

import torch
import json
import pandas as pd

from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler, Subset

from bertron.train import input_dict, evaluate, prepare_train, tokenize_data
from bertron.utils import be_there_gpus, f1_score_func, accuracy_per_class

seed = 666
DEVICE = be_there_gpus()


def load_model():
    """

    :return:
    """

    with open('data/processed/labels.json', 'r') as f_in:
        label_dict = json.load(f_in)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_dict),
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(DEVICE)

    # load transfer learning component for Enron corpus
    checkpoint = torch.load('bertron/models/finetuned_BERT_epoch_1.model')
    model.load_state_dict(checkpoint)
    model.eval()

    return model, label_dict


def predict(data, model, num_records=64, batch_size=16):
    """

    :param data:
    :param model:
    :param num_records:
    :param batch_size:
    :return:
    """

    X, max_len = prepare_train(data)
    _, dataset_test = tokenize_data(X, max_len)
    dataloader_test = DataLoader(
        dataset_test,
        sampler=SequentialSampler(dataset_test),
        batch_size=batch_size
    )

    # generate predictions for only a small number
    indices = torch.arange(num_records)
    dataloader_predict = Subset(dataloader_test, indices)
    _, predictions, truths = evaluate(model, dataloader_predict)

    # softmax?
    predictions = [p.argmax() for p in predictions[0]]

    return predictions, truths


def main():
    """

    :return:
    """

    # load model
    model, label_dict = load_model()

    # load scoring data
    # really ought to have a proper validation holdout set
    # until then, use test dataset (at least not seen during training)
    X = pd.read_parquet('data/processed/X.parquet').sample(frac=.05)

    predictions, truths = predict(X, model)
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    # how'd we do?
    print(f'true authors     : {[label_dict_inverse[x] for x in truths]}')
    print(f'predicted authors: {[label_dict_inverse[x] for x in predictions]}')
    print(f'f1 score: {f1_score_func(predictions, truths)}')
    accuracy_per_class(predictions, truths)  # this already prints


if __name__ == '__main__':
    main()
