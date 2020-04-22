import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

##########################################################################################
# adversarial siamese network #train
class adv_SiameseTrainData(Dataset):
    def __init__(self, train_A_datasets,train_B_datasets):
        self.train_A_datasets = train_A_datasets
        self.train_B_datasets = train_B_datasets

        self.transform = self.train_A_datasets.transform

        self.train_labels_num = len(self.train_A_datasets.classes)
        self.train_labels_set = set(range(self.train_labels_num))

        self.train_A_data = self.train_A_datasets.imgs
        train_A_datanum = len(self.train_A_data)
        self.train_A_img = []
        self.train_A_label = np.zeros(train_A_datanum)
        for i in range(train_A_datanum):
            self.train_A_img.append(self.train_A_data[i][0])
            self.train_A_label[i] = self.train_A_data[i][1]

        self.train_B_data = self.train_B_datasets.imgs
        train_B_datanum = len(self.train_B_data)
        self.train_B_img = []
        self.train_B_label = np.zeros(train_B_datanum)
        for i in range(train_B_datanum):
            self.train_B_img.append(self.train_B_data[i][0])
            self.train_B_label[i] = self.train_B_data[i][1]

        self.B_label_to_indices = {label: np.where(self.train_B_label == label)[0]
                                     for label in self.train_labels_set}

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        img1, label1 = self.train_A_img[index], self.train_A_label[index]
        if target == 1:
            siamese_index = np.random.choice(self.B_label_to_indices[label1])
            label2 = label1
        else:
            siamese_label = np.random.choice(list(self.train_labels_set - set([label1])))
            siamese_index = np.random.choice(self.B_label_to_indices[siamese_label])
            label2 = siamese_label
        img2 = self.train_B_img[siamese_index]
        
        if self.transform is not None:
            img1 = Image.open(img1)
            img1 = self.transform(img1)
            img2 = Image.open(img2)
            img2 = self.transform(img2)
        return (img1, img2), target, label1, label2

    def __len__(self):
        return len(self.train_A_datasets)
