import json
import random
from os import makedirs

data = json.load(open('scraped_data/data_final_1024.json'))
random.shuffle(data)

num_train = int(0.8 * len(data))
num_val = int(0.1 * len(data))

split_data = {
    'train': data[:num_train],
    'val': data[num_train:num_train+num_val],
    'test': data[num_train+num_val:]
}

makedirs('scraped_data/data-1024')
for split in ['train', 'val', 'test']:
    doi_file = open(f'scraped_data/data-1024/{split}.doi', 'w')
    source_file = open(f'scraped_data/data-1024/{split}.source', 'w')
    target_file = open(f'scraped_data/data-1024/{split}.target', 'w')

    for article in split_data[split]:
        doi, abstract, pls = article['doi'], article['abstract'], article['pls']
        doi_file.write(doi + '\n')
        source_file.write(abstract + '\n')
        target_file.write(pls + '\n')

    doi_file.close(), source_file.close(), target_file.close()

