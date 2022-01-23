from torch.utils.data import Dataset
import numpy as np

class KitchenDataset(Dataset):
    def __init__(self, file_name):
        super().__init__()
        dataset = np.load(file_name, allow_pickle=True).item()
        self.data = {}

        self.data['obs'] = dataset['observations'][:, 0:30]
        self.data['act'] = dataset['actions']
        self.length = self.data['obs'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data['obs'][index,:], self.data['act'][index,:]
