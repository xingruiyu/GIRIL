from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import pickle
import torch

class ExpertDataset(torch.utils.data.TensorDataset):
    """

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        action_transform (callable, optional): A function/transform that takes in the
            action and transforms it.
    """
    def __init__(self, file_name, num_trajectories, train=True, train_test_split=1.0, \
        subsample_frequency=4, transform=None, action_transform=None, download=False, return_next_state=False):

        all_trajectories = pickle.load(open(file_name, 'rb'))
        print('loaded demonstrations with returns:', [r.item() for r in all_trajectories['returns']])
        print('loaded demonstrations with lengths:', all_trajectories['lengths'])

        self.transform = transform
        self.action_transform = action_transform
        self.train = train  # training set or test set
        self.split = train_test_split
        self.return_next_state = return_next_state

        acc_eps_lengths = []
        length = 0
        for l in all_trajectories['lengths']:
            length += l
            acc_eps_lengths.append(length)
        
        idx = acc_eps_lengths[num_trajectories-1]

        start_idx = torch.randint(
            0, subsample_frequency, size=(1, )).long()

        self.trajectories = {}

        for k, v in all_trajectories.items():
            if k == 'states':
                state_data = v[:idx-1]
                next_state_data = v[1:idx]
                self.trajectories['states'] = torch.from_numpy(state_data[start_idx::subsample_frequency]).float()
                self.trajectories['next_states'] = torch.from_numpy(next_state_data[start_idx::subsample_frequency]).float()
            elif k in ['actions', 'rewards']: 
                data = v[:idx-1]
                self.trajectories[k] = torch.from_numpy(data[::subsample_frequency]).float()
            else:
                data = all_trajectories[k][:num_trajectories]
                self.trajectories[k] = torch.from_numpy(data).float()

        self.length = self.trajectories['states'].shape[0]

        if self.train:
            self.train_states = self.trajectories['states'][:int(self.length*self.split)]
            self.train_actions = self.trajectories['actions'][:int(self.length*self.split)]
            self.train_next_states = self.trajectories['next_states'][:int(self.length*self.split)]
            print('Total training states: %s' %(self.train_states.shape[0]))
        else:
            self.test_states = self.trajectories['states'][int(self.length*self.split):]
            self.test_actions = self.trajectories['actions'][int(self.length*self.split):]
            self.test_next_states = self.trajectories['next_states'][int(self.length*self.split):]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (state, action) where action is index of the action class.
        """
        if self.train:
            state, action, next_state = self.train_states[index], self.train_actions[index], self.train_next_states[index]
        else:
            state, action, next_state = self.test_states[index], self.test_actions[index], self.test_next_states[index]

        if self.return_next_state:
            return state, action, next_state
        else:
            return state, action

    def __len__(self):
        if self.train:
            return len(self.train_states)
        else:
            return len(self.test_states)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.action_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


