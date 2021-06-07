import json
import os
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import BartTokenizer
import sys

def abs_length(article):
    return sum([len(x['text']) for x in article['abstract']])

def pls_length(article):
    if article['pls_type'] == 'long':
        return len(article['pls'])
    else:
        return sum([len(x['text']) for x in article['pls']])

def res_para(text):
    first_index = -1
    sentence_list = sent_tokenize(text)
    for index, sentence in enumerate(sentence_list):
        if any(word in sentence.lower() for word in ['journal', 'study', 'studies', 'trial']):
            first_index = index
            break
    return first_index > -1 and (index+1)/len(sentence_list) <= 0.5

def res_heading(heading):
    return any(word in heading.lower() for word in ['find', 'found', 'evidence', 'tell us', 'study characteristic'])

def one_para_filter(text):
    sentence_list = sent_tokenize(text)
    first_index = -1
    for index, sentence in enumerate(sentence_list):
        if any(word in sentence.lower() for word in ['review', 'journal', 'study', 'studies', 'paper', 'trial']):
            first_index = index
            break
    return ' '.join(sentence_list[first_index:]) if first_index > -1 else ''

def clean_up_data(fname):
    data = json.load(open(fname))

    #truncate abstract to only main results onwards
    for article in data:
        first_index = -1
        for index,section in enumerate(article['abstract']):
            if 'main result' in section['heading'].strip().lower():
                first_index = index
                break
        article['abstract'] = article['abstract'][first_index:]

    #get rid of reviews with character counts < 1000
    data = [x for x in data if abs_length(x) >= 1000]

    #split the data into long and sectioned parts
    data_long = [x for x in data if x['pls_type']=='long']
    data_sectioned = [x for x in data if x['pls_type']=='sectioned']

    #now split long into 1-paragraph and multi-paragraph
    data_long_single = [x for x in data_long if len(x['pls'].strip().split('\n'))==1]
    data_long_multi = [x for x in data_long if len(x['pls'].strip().split('\n')) > 1]

    #truncate all the reviews' plain-language summary appropriately
    for article in data_long_single:
        article['pls'] = one_para_filter(article['pls'])
    
    for article in data_long_multi:

        first_index = -1
        for index,para in enumerate(article['pls'].strip().split('\n')):
            if res_para(para):
                first_index = index
                break
        
        if first_index > -1:
            article['pls'] = '\n'.join(article['pls'].strip().split('\n')[first_index:])
        else:
            article['pls'] = ''
    
    data_long_single = [x for x in data_long_single if len(x['pls']) > 0]
    data_long_multi = [x for x in data_long_multi if len(x['pls']) > 0]
    
    for article in data_sectioned:
        first_index = -1
        for index,section in enumerate(article['pls']):
            if res_heading(section['heading']):
                first_index = index
                break
        
        if first_index > -1:
            article['pls'] = article['pls'][first_index:]
        else:
            article['pls'] = []
    
    data_sectioned = [x for x in data_sectioned if len(x['pls']) > 0]

    #now trim based on ratio of pls length to abstract length
    data_long_single = [x for x in data_long_single if (pls_length(x)/abs_length(x) >= 0.20 and pls_length(x)/abs_length(x) <= 1.4)]
    data_long_multi = [x for x in data_long_multi if (pls_length(x)/abs_length(x) >= 0.30 and pls_length(x)/abs_length(x) <= 1.3)]
    data_sectioned = [x for x in data_sectioned if (pls_length(x)/abs_length(x) >= 0.30 and pls_length(x)/abs_length(x) <= 1.3)]
    data_final = data_long_single+data_long_multi+data_sectioned

    #now only keep the articles in which both the abstract and PLS have <= 1024 tokens
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
    data_final_1024 = []
    for article in data_final:
        abstract = article['abstract']
        pls = article['pls']
        if len(tokenizer(abstract)['input_ids']) <= 1024 and len(tokenizer(pls)['input_ids']) <= 1024:
            data_final_1024.append(article)

    with open('scraped_data/data_final.json', 'w') as f:
        f.write(json.dumps(data_final, indent=2))

    with open('scraped_data/data_final_1024.json', 'w') as f:
        f.write(json.dumps(data_final_1024, indent=2))

def main():
    clean_up_data('scraped_data/data.json')

if __name__ == "__main__":
    main()