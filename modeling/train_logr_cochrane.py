import json
from joblib import dump
from os.path import join
from random import shuffle
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from transformers import BartTokenizer

def get_abstract(article):
    return ' '.join([x['text'] for x in article['abstract']])

def get_pls(article):
    return article['pls'] if article['pls_type'] == 'long' else ' '.join([x['text'] for x in article['pls']])

def make_vector(text, tokenizer):
    token_ids = tokenizer.encode(text)[1:-1]
    count_vector = np.zeros(tokenizer.vocab_size, dtype=np.int16)
    for ID in token_ids:
        count_vector[ID] += 1
    return count_vector

def construct_dataset(data, tokenizer):
    data = json.load(open(data)) if type(data)==str else data
    shuffle(data)

    X = np.empty((2*len(data), tokenizer.vocab_size), dtype=np.int16)
    y = np.empty(2*len(data), dtype=np.int16)

    index = 0
    for article in data:
        X[index] = make_vector(article["abstract"], tokenizer)
        X[index+1] = make_vector(article["pls"], tokenizer)
        y[index] = 0
        y[index+1] = 1
        index += 2

    return X, y

def get_vocab(tokenizer):
    tokens = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(tokenizer.vocab_size)]
    return tokens

def logr_simple_term_counts(tokenizer, save_fname, data_dir='data/data_final.json', weights_dir='data/logr_weights'):

    X_train, y_train = construct_dataset(data_dir, tokenizer)

    #apply feature scaling
    X_train = normalize(X_train)

    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)

    dump(model, save_fname)

    vocab = get_vocab(tokenizer)
    weights = np.squeeze(model.coef_, axis=0).tolist()

    sorted_weights = filter(lambda x: len(x[1].strip()) > 0, zip(range(tokenizer.vocab_size), vocab, weights))
    sorted_weights = list(sorted(sorted_weights, key=lambda x: x[2]))

    with open(join(weights_dir, 'bart_freq_normalized_ids.txt'), 'w') as f:
        for ID, word, weight in sorted_weights:
            f.write(f'{ID} {weight}\n')

    with open(join(weights_dir, 'bart_freq_normalized_tokens.txt'), 'w') as f:
        for ID, word, weight in sorted_weights:
            f.write(f'{word} {weight}\n')

def list_index(l, indices):
    return [l[i] for i in indices]

def simple_kfold_term_counts(tokenizer, data_dir='data/data_final.json', k=5):

    X, y = construct_dataset(data_dir, tokenizer)
    splitter = StratifiedKFold(n_splits=k, shuffle=True)
    accuracies = np.zeros(k)

    for i,(train_indices, test_indices) in enumerate(splitter.split(X, y)):

        print(f'beginning fold {i}')

        train_indices = train_indices.tolist()
        test_indices = test_indices.tolist()

        X_train = list_index(X, train_indices)
        y_train = list_index(y, train_indices)
        X_test = list_index(X, test_indices)
        y_test = list_index(y, test_indices)

        #apply feature scaling
        X_train = normalize(X_train)
        X_test = normalize(X_test)

        model = LogisticRegression(max_iter=100)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracies[i] = accuracy_score(y_test, predictions)

    print(np.mean(accuracies))

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
# logr_simple_term_counts(tokenizer, 
#                         save_fname='data/logr_model/model.joblib', 
#                         data_dir='data/data_final_1024.json', 
#                         weights_dir='data/logr_weights')

# simple_kfold_term_counts(tokenizer, data_dir='data/data_final_1024.json', k=5)