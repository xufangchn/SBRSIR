from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import os
import numpy as np

# data folder
data_dir = "../data/a"

# Number of categories
num_category = 15

# Batch size for training
batch_size = 50

# Number of epochs to train for 
num_epochs = 40

# the size of input image
input_size = 224

# save model 
saving_model_name = '../models/rsketch'

# define data_transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load Train Data
train_A_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train','sketch'), data_transforms['train']) 
train_B_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train','image'), data_transforms['train']) 

from datasetsbuilding import adv_SiameseTrainData
siamese_train_dataset = adv_SiameseTrainData(train_A_datasets,train_B_datasets) # Returns pairs of images and target same/different
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

# Load Test Data
test_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, 'test', x), data_transforms['test']) for x in ['query', 'database']}
test_loader = {x: torch.utils.data.DataLoader(test_dataset[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in ['query', 'database']}

##########################################################################################
# Set up the network and training parameters
import mymodel

model_pretrained = models.resnet50(pretrained=False)                     # backbone
model_pretrained.load_state_dict(torch.load('resnet50-19c8e357.pth'))    # weights of imagenet
num_ftrs = model_pretrained.fc.in_features                               # fc_in
model_pretrained.fc = nn.Linear(num_ftrs, num_category)                  # modify fc_out
num_cls = model_pretrained.fc.out_features                               # fc_out
model = mymodel.myresnet50(model_pretrained, num_cls)                    # load my model
model_dis = mymodel.Discriminator()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model_dis.to(device)

from trainer import set_parameter_requires_grad
set_parameter_requires_grad(model)
params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

from losses import ContrastiveLoss
margin = 1.
criterion_sia = ContrastiveLoss(margin)
criterion_ft = nn.CrossEntropyLoss()
criterion_dis = nn.BCELoss()

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
optimizer_dis = optim.Adam(model_dis.parameters(), lr=0.002, betas=(0.5, 0.999))

scheduler_ft = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

from trainer import adv_fit
adv_fit(siamese_train_loader, model, model_dis, saving_model_name, device, test_loader, criterion_sia, criterion_ft, criterion_dis, optimizer_ft, optimizer_dis, scheduler_ft, num_epochs)