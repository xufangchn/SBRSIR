from __future__ import print_function 
from __future__ import division
import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import torch.nn as nn

def adv_acc_compute(model_ft,device,dataloaders_dict):
    unseen_labels = [0, 4, 8, 12, 16]
    # unseen_labels = [1, 5, 9, 13, 17]
    # unseen_labels = [2, 6, 10, 14, 18]
    # unseen_labels = [3, 7, 11, 15, 19]

    # Iterate over data.
    query_num=0
    database_num=0

    # extract query feature
    for inputs, labels in dataloaders_dict['query']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = resnet_forward(model_ft,inputs)
        
        if query_num == 0:
            query_feature = outputs
            query_labels = labels
            query_inputs = inputs.cpu()
        else:
            query_feature=torch.cat([query_feature, outputs],0)
            query_labels=torch.cat([query_labels, labels],0)
            query_inputs=torch.cat([query_inputs, inputs.cpu()],0)
        query_num = query_num + inputs.size(0)

    query_feature=query_feature.cpu().numpy()
    query_labels=query_labels.cpu().numpy()

    # extract database feature
    for inputs, labels in dataloaders_dict['database']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = resnet_forward(model_ft,inputs)

        if database_num == 0:
            database_feature = outputs
            database_labels = labels
            database_inputs = inputs.cpu()
        else:
            database_feature=torch.cat([database_feature, outputs],0)
            database_labels=torch.cat([database_labels, labels],0)
            database_inputs=torch.cat([database_inputs, inputs.cpu()],0)
        database_num = database_num + inputs.size(0)

    database_feature=database_feature.cpu().numpy()
    database_labels=database_labels.cpu().numpy()

    AP = np.array([])#the average precision 
    top10 = np.array([])
    top50 = np.array([])
    top100 = np.array([])

    seen_AP = np.array([])
    seen_top10 = np.array([])
    seen_top50 = np.array([])
    seen_top100 = np.array([])

    unseen_AP = np.array([])
    unseen_top10 = np.array([])
    unseen_top50 = np.array([])
    unseen_top100 = np.array([])

    for i in range(query_num):
        q_feature = query_feature[i,:]#the ith query image feature
        q_label = query_labels[i]#the ith query image label

        if q_label in unseen_labels:
            unseen = 1
        else:
            unseen = 0

        rsum2_fea_norm = []

        for hh in range(len(database_feature)):
            d_feature = database_feature[hh,:]
            rsum2_fea_norm = np.append(rsum2_fea_norm, np.linalg.norm( q_feature - d_feature ))

        index = np.argsort(rsum2_fea_norm)

        true_matches = 0#the num of true_matches in retrieval for the ith query image feature
        TRUE_matches = np.sum(database_labels == q_label)#the num of ground truth for the ith query image feature

        j = 0#the jth return image for the ith query image feature
        p = np.array([])#precision array for the ith query image feature
        for h in index:
            j = j+1#the jth return image
            dt_label = database_labels[h]#the jth return image label from database
            if q_label == dt_label:#true retrieval
                true_matches = true_matches + 1
                pp = true_matches/j
                p = np.append(p,pp)
            
            if j == 10:
                t10 = true_matches/10
                top10 = np.append(top10,t10)
                if unseen == 1:
                    unseen_top10 = np.append(unseen_top10,t10)
                else:
                    seen_top10 = np.append(seen_top10,t10) 
            elif j ==50:
                t50 = true_matches/50
                top50 = np.append(top50,t50)  
                if unseen == 1:
                    unseen_top50 = np.append(unseen_top50,t50)
                else:
                    seen_top50 = np.append(seen_top50,t50)    
            elif j==100:
                t100 = true_matches/100
                top100 = np.append(top100,t100)
                if unseen == 1:
                    unseen_top100 = np.append(unseen_top100,t100)
                else:
                    seen_top100 = np.append(seen_top100,t100)
            if true_matches == TRUE_matches:# retrieval finished
                ap = np.mean(p)
                AP = np.append(AP,ap)
                if unseen == 1:
                    unseen_AP = np.append(unseen_AP,ap)
                else:
                    seen_AP = np.append(seen_AP,ap)
                break

    # for sum
    mAP = np.mean(AP)
    TOP10 = np.mean(top10)
    TOP50 = np.mean(top50)
    TOP100 = np.mean(top100)
    print('{} mAP: {:.4f} TOP10: {:.4f} TOP50: {:.4f} TOP100: {:.4f}'.format('sum', mAP, TOP10, TOP50, TOP100))

    # for unseen
    unseen_mAP = np.mean(unseen_AP)
    unseen_TOP10 = np.mean(unseen_top10)
    unseen_TOP50 = np.mean(unseen_top50)
    unseen_TOP100 = np.mean(unseen_top100)
    print('{} mAP: {:.4f} TOP10: {:.4f} TOP50: {:.4f} TOP100: {:.4f}'.format('unseen', unseen_mAP, unseen_TOP10, unseen_TOP50, unseen_TOP100))

    # for seen
    seen_mAP = np.mean(seen_AP)
    seen_TOP10 = np.mean(seen_top10)
    seen_TOP50 = np.mean(seen_top50)
    seen_TOP100 = np.mean(seen_top100)                           
    print('{} mAP: {:.4f} TOP10: {:.4f} TOP50: {:.4f} TOP100: {:.4f}'.format('seen', seen_mAP, seen_TOP10, seen_TOP50, seen_TOP100))


######################################################################
def resnet_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)

    return x