#!/usr/bin/env python
# coding: utf-8

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import transformers

from tqdm import tqdm

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import torch.multiprocessing as mp
mp.set_start_method('fork', force=True)

class NERDataset(Dataset):
    
    def __init__(self, filename, label2id):
        
        self.sentences = []
        with open(filename, 'r') as f:
            sentence = []
            for l in f:
                if l == '\n':
                    self.sentences.append(zip(*sentence))
                    sentence = []
                else:
                    token, cls = l.strip().split('\t')
                    sentence.append((token, label2id[cls]))
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx):
        return self.sentences[idx]

def dl_collate_fn(data):
    return tuple(zip(*data))

# token alignment
def tokenize_and_align_labels(tokenizer, batch):

    tokenized_inputs = tokenizer(batch[0], padding=True, truncation=True, is_split_into_words=True, return_tensors='pt')
    labels = []

    for i, label in enumerate(batch[1]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = torch.tensor(labels)

    return tokenized_inputs

if __name__ == "__main__":
    lr = 1e-3
    
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    
    label2id = {'O': 0,
     'B-Biological_Molecule': 1,
     'E-Biological_Molecule': 2,
     'S-Biological_Molecule': 3,
     'I-Biological_Molecule': 4,
     'S-Species': 5,
     'B-Species': 6,
     'I-Species': 7,
     'E-Species': 8,
     'B-Chemical_Compound': 9,
     'E-Chemical_Compound': 10,
     'S-Chemical_Compound': 11,
     'I-Chemical_Compound': 12}
    
    id2label = {v: k for k, v in label2id.items()}
    
    train_ds = NERDataset('train.txt', label2id)
    val_ds = NERDataset('dev.txt', label2id)
    
    train_dl = DataLoader(train_ds, batch_size=8, collate_fn=dl_collate_fn, num_workers=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=8, collate_fn=dl_collate_fn, num_workers=8, shuffle=False)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-cased')
    model = transformers.AutoModelForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=13, id2label=id2label, label2id=label2id)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    
    for epoch in range(2):
    
        print(f'Epoch {epoch+1}:')
        train_loss = 0
        model.eval()
        for batch in tqdm(train_dl):
            tok_batch = tokenize_and_align_labels(tokenizer, batch)
            tok_batch = {k : v.to(device) for k, v in tok_batch.items()}
    
            # optimizer.zero_grad()
            loss = model(**tok_batch).loss
            # loss.backward()
            # optimizer.step()
    
            train_loss += loss.detach()
    
        train_loss = train_loss.cpu()
        train_loss /= len(train_dl)
        print(f' Train Loss: {train_loss}')
    
        val_loss = 0
        model.eval()
        for batch in tqdm(val_dl):
            tok_batch = tokenize_and_align_labels(tokenizer, batch)
            tok_batch = {k : v.to(device) for k, v in tok_batch.items()}
    
            loss = model(**tok_batch).loss
            val_loss += loss.detach()
    
        val_loss = val_loss.cpu()
        val_loss /= len(val_dl)
        print(f' Val Loss: {val_loss}')
        print('')
