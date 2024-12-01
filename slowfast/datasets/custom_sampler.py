import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler

class FewShotEpisodeSampler(Sampler):
    def __init__(self, dataset, cfg, mode, less_iters=False):
        self.cfg = cfg
        random.seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)

        self.mode = mode
        labels = dataset._labels
        self.class_ids = list(np.unique(labels))
        self.num_way = cfg.FEW_SHOT.N_WAY
        self.num_support = cfg.FEW_SHOT.K_SHOT 
        self.num_queries = (cfg.FEW_SHOT.TRAIN_QUERY_PER_CLASS if mode == 'train' 
                                            else cfg.FEW_SHOT.TEST_QUERY_PER_CLASS)
        self.samples_per_class = self.num_support + self.num_queries
        self.batch_size = (self.num_way * self.samples_per_class)

        # Create a list of indices for each class
        self.class_indices = {class_label: [idx for idx, (label) in enumerate(labels) if label == class_label] 
                              for class_label in self.class_ids}
        self.less_iters = less_iters

    def __iter__(self):
        while True:
            selected_classes = random.sample(self.class_ids, self.num_way)

            batch_indices = []
            sample_types = []
            batch_label = []

            sample_type = (['support'] * self.num_support + 
                                            ['query'] * self.num_queries)
            for idx, class_label in enumerate(selected_classes):
                # Sample 'samples_per_class' indices from each selected class
                class_indices = random.sample(self.class_indices[class_label], 
                                                        self.samples_per_class)
                batch_indices.extend(class_indices)
                sample_types.extend(sample_type)
                batch_label.extend([idx] * self.samples_per_class)
            batch_indices = np.array(batch_indices)
            sample_types = np.array(sample_types)
            batch_label = np.array(batch_label)
            indices = list(range(len(batch_indices)))

            # Shuffle the batch indices to mix the classes
            random.shuffle(indices)
            batch_indices = batch_indices[indices]
            sample_types = sample_types[indices] 
            batch_label = batch_label[indices]
            index_and_sample_info = list(zip(batch_indices, batch_label, sample_types))

            # Yield batches of size 'batch_size'
            for i in range(0, len(batch_indices), self.batch_size):
                yield index_and_sample_info[i:i+self.batch_size]
    def __len__(self):
        div_factor = self.cfg.NUM_GPUS if not self.cfg.FEW_SHOT.TRAIN_OG_EPISODES else 1
        if self.mode == 'train':
            return self.cfg.FEW_SHOT.TRAIN_EPISODES // div_factor
        else:
            if self.less_iters:
                return self.cfg.FEW_SHOT.TEST_EPISODES // div_factor // 5
            return self.cfg.FEW_SHOT.TEST_EPISODES // self.cfg.NUM_GPUS 
        
