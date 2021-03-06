import glob
from pprint import pprint
import pandas as pd
import numpy as np
import json


def format_email(data, fields=['To', 'X-To', 'From', 'X-From', 'cc', 'X-cc', 'Subject', 'Body']):
    """ Given a plain text email file, return a dictionary of key:value pairs according to
    fields specified in function arg call

    :param data: str
        contents of emails from Enron corpus, access via file.read()
    :param fields: list
        list of strings from available email metadata included as plain text

    :return: dict
        dictionary of key:value pairs according to fields specified in args
    """

    d = {}

    lines = data.split('\n')
    for i in range(0, len(lines)):

        # header info ends with blank line
        if lines[i] != '':
            try:
                # field and value delimited with :
                key, value = lines[i].split(':', maxsplit=1)
            except:
                try:
                    # sometimes line continuations
                    value += lines[i].split('\t', maxsplit=1)[1]
                except:
                    pass

            # add entry to dict
            d[key] = value
        else:
            break

    key = 'Body'
    value = []
    # after header is text body, skip empty line
    for j in range(i + 1, len(lines)):

        # anything below dashes are forwards/replies, don't include
        if not ((lines[j].startswith(' -----')) or
                (lines[j].startswith('-----'))):

            # append each line
            value.append(lines[j])
        else:
            break

    # preserve original white space
    d[key] = '\n'.join(value)

    # only return certain header info specified in function args
    return {key: d[key] for key in fields if key in d}


def format_raw(subdir):
    """ Within the Enron corpus, specify a type of email from among possible subdirectories,
    and return a pandas dataframe of email subfields

    :param subdir: str
        subdirectory of email files

    :return: pandas.DataFrame
        dataframe with each record being an email, and each column being a field specified in
        the format_email() invocation
    """
    f_paths = []

    # use glob to search for all sent items
    for f_name in glob.glob(f'..\data\maildir\*\*{subdir}*\*'):
        # some weird windows thing, the slashes are all the wrong way
        f_paths.append(f_name.replace('\\', '/'))

    print(f'number of sent items: {len(f_paths)}')

    # main invocation
    emails = []
    for f_path in f_paths:
        try:
            with open(f_path, 'r') as f:
                # loop through glob filepaths and append to list of dicts
                emails.append(format_email(f.read()))
        except:
            pass

    print(f'emails processed: {len(emails)}')

    return pd.DataFrame(emails).dropna()


def format_names(x):
    """ Given a string of recipients, extract the username for each

    :param x: str
        string of email recipients (comma-delimited)

    :return: list
        list of strings, with each string being the username (preceding @ symbol)
    """
    return [y.split('@')[0] for y in x.strip().split(',')]


def assign_labels(X, proportion=0.67):
    """

    :param X: pandas.DataFrame
        email dataframe containing 'author' field
    :param proportion: float
        float between 0-1: proportion of emails to represent in data set by top n
        authors by number of sent emails

    :return: (pandas.DataFrame, dict)
        email dataframe with numerical labels for each author
        dict mapping numerical labels to author
    """

    vc = X['author'].value_counts()

    # how many authors (sorted descending) account for proportion=0.67 of emails
    # answer, about 33
    # only include these in training data (no long tail)
    cs = vc.cumsum() < vc.sum() * proportion
    authors = vc[cs]

    # filter to most commonly occurring authors
    mask = X['author'].apply(lambda x: x in authors.index)
    X = X[mask]

    # encode authors with numerical labels
    label_dict = {}
    for index, label in enumerate(authors.index):
        label_dict[label] = index

    X = X.assign(label=X['author'].replace(label_dict))

    return X, label_dict


def format_train(df, subset_frac=1.):
    """ Given a dataframe of all possible email fields, format for author
    assignment modeling

    :param df: pandas.DataFrame
        full email dataframe (output of format_raw())
    :param subset_frac: float
        between 0-1, proportion of email records to return (smaller values for
        testing pipeline)

    :return: pandas.DataFrame
        dataframe containing ['author', 'recip_primary', 'subject', 'body'] fields
    """

    df = df.assign(recip=df['To'].apply(format_names),
                   author=df['From'].apply(lambda x: x.strip().split('@')[0]))
    # wish this worked
    # df = df.assign(recip, author = map(lambda col: df[col].apply(format_names), ['To', 'From']))

    # for this purpose, only concerned with primary (first) recipient
    df = df.assign(recip_primary=df['recip'].apply(lambda x: x[0]))
    df[['subject', 'body']] = df[['Subject', 'Body']].apply(lambda col: col.str.strip())

    #  limited X (train) dataframe
    X = df[['author', 'recip_primary', 'subject', 'body']].dropna()
    X = X.replace('', np.nan).dropna()

    # subset for pipeline evaluation purposes
    X = X.sample(frac=subset_frac)
    assert X.isna().sum().sum() == 0

    return X


def main():
    """ main invocation
    """

    # read and format raw sent email files
    # this is pretty slow, done serially
    df = format_raw('sent')

    # write to disk
    df.to_parquet('../data/processed/emails.parquet')
    # df = pd.read_parquet('../data/processed/emails.parquet')

    # prepare for training
    X = format_train(df, subset_frac=1)
    X, label_dict = assign_labels(X)

    # write train-processed data to disk
    X.to_parquet('../data/processed/X.parquet')
    with open('../data/processed/labels.json', 'w') as f_out:
        json.dump(label_dict, f_out)

    print(X.shape)
    pprint(label_dict)


if __name__ == '__main__':
    main()
