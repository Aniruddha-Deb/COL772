from torch.utils.data import Dataset

class BioNERDataset(Dataset):
    
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
