import torch
import h5py, mne, os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.distributed import DistributedSampler



class cock_tail(Dataset):
    def __init__(self, root, mode, subject=None):
        super(cock_tail).__init__()
        self.root = root
        if mode == 'train':
            self.file_path = os.path.join(root, 'Cocktail_Party/Normalized/2s/fbc/new') 
            self.noisy_file = os.path.join(self.file_path, 'noisy_train.h5')
            self.clean_file = os.path.join(self.file_path, 'clean_train.h5')
            self.eeg_file = os.path.join(self.file_path, 'eegs_train.h5')
            f = h5py.File(self.noisy_file, 'r')
            d = f['noisy_train=']
            self.mode = mode
            self.length = len(d)
            f.close()
        elif mode == 'val':
            self.file_path = os.path.join(root, 'Cocktail_Party/Normalized/2s/fbc/new')
            self.noisy_file = os.path.join(self.file_path, 'noisy_val.h5')
            self.clean_file = os.path.join(self.file_path, 'clean_val.h5')
            self.eeg_file = os.path.join(self.file_path, 'eegs_val.h5')
            f = h5py.File(self.noisy_file, 'r')
            d = f['noisy_val=']
            self.mode = mode
            self.length = len(d)
            f.close()
        else:
            self.subject = subject
            self.file_path = os.path.join(root, 'Cocktail_Party/Normalized/20s/fbc/new')
            self.noisy_file = os.path.join(self.file_path, 'noisy_test.h5')
            self.clean_file = os.path.join(self.file_path, 'clean_test.h5')
            self.eeg_file = os.path.join(self.file_path, 'eegs_test.h5')
            self.subject_file = os.path.join(self.file_path, 'subjects_test.h5')
            self.mode = mode
            if subject != None:
                subject_f = h5py.File(self.subject_file, 'r')
                subject_key = [key for key in subject_f][0]
                self.samples_of_interest = [i for i, s in enumerate(subject_f[subject_key][:]) if s == subject]
                self.length = len(self.samples_of_interest)
                subject_f.close()
            else:
                f = h5py.File(self.noisy_file, 'r')
                d = f['noisy_test=']
                self.length = len(d)
                f.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode != 'test':
            nosiy_f = h5py.File(self.noisy_file, 'r')
            clean_f = h5py.File(self.clean_file, 'r')
            eeg_f = h5py.File(self.eeg_file, 'r')
            noisy_data = nosiy_f['noisy_{}='.format(self.mode)]
            clean_data = clean_f['clean_{}='.format(self.mode)]
            eeg_data = eeg_f['eegs_{}='.format(self.mode)]
            n_d = np.transpose(noisy_data[idx])
            n_d = (n_d[:, ::3]).astype(np.float32)
            e_d = np.transpose((eeg_data[idx]).squeeze())
            e_d = (np.repeat(e_d, 114, 1)).astype(np.float32)
            c_d = np.transpose(clean_data[idx])
            c_d = (c_d[:, ::3]).astype(np.float32)
            n_d = torch.tensor(n_d)
            e_d = torch.tensor(e_d)
            c_d = torch.tensor(c_d)
            nosiy_f.close()
            clean_f.close()
            eeg_f.close()
        else:
            true_idx = idx
            if self.subject == None:
                true_idx = idx
            else:
                true_idx = self.samples_of_interest[idx]
            nosiy_f = h5py.File(self.noisy_file, 'r')
            clean_f = h5py.File(self.clean_file, 'r')
            eeg_f = h5py.File(self.eeg_file, 'r')
            noisy_data = nosiy_f['noisy_{}='.format(self.mode)]
            clean_data = clean_f['clean_{}='.format(self.mode)]
            eeg_data = eeg_f['eegs_{}='.format(self.mode)]
            n_d = np.transpose(noisy_data[true_idx])
            n_d = (n_d[:, ::3]).astype(np.float32)
            e_d = np.transpose((eeg_data[true_idx]).squeeze())
            e_d = (np.repeat(e_d, 114, 1)).astype(np.float32)
            c_d = np.transpose(clean_data[true_idx])
            c_d = (c_d[:, ::3]).astype(np.float32)
            n_d = torch.tensor(n_d)
            e_d = torch.tensor(e_d)
            c_d = torch.tensor(c_d)
            nosiy_f.close()
            clean_f.close()
            eeg_f.close()
        return n_d, e_d, c_d

def load_CleanNoisyPairDataset(root, subset, batch_size, num_gpus=1):
    """
    Get dataloader with distributed sampling
    """
    dataset = cock_tail(root=root, mode=subset)
    kwargs = {"batch_size": batch_size, "num_workers": 8, "pin_memory": True, "drop_last": False}
    if num_gpus > 1:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
    else:
        if subset == 'train':
            dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=True, **kwargs)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=False, **kwargs)

    return dataloader