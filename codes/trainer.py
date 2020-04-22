import torch
import numpy as np
import copy
from acc import adv_acc_compute
import matplotlib.pyplot as plt 

A_label = 1
B_label = 0

lambda_cls = 1
lambda_sia = 0.01
lambda_dm = 0.01

##########################################################################################
def adv_fit(train_loader, model, model_dis, saving_model_name, device, test_loader, criterion_sia, criterion_ft, criterion_dis, optimizer_ft, optimizer_dis, scheduler_ft, n_epochs):

    # loss curve
    loss_L = np.array([])
    loss_Ldis = np.array([])
    
    for epoch in range(n_epochs):

        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        # Train stage
        loss_L, loss_Ldis = adv_train_epoch(train_loader, model, model_dis, criterion_sia, criterion_ft, criterion_dis, optimizer_ft, optimizer_dis, device, loss_L, loss_Ldis)
        # Test stage
        adv_test_epoch(test_loader, model, device)
        # save model
        save_model(epoch, saving_model_name, model)
        # adjust learning rate
        scheduler_ft.step()

##########################################################################################
# Adversarial Train stage
def adv_train_epoch(train_loader, model, model_dis, criterion_sia, criterion_ft, criterion_dis, optimizer_ft, optimizer_dis, device, loss_L, loss_Ldis):

    # adjust learning rate

    running_loss0 = 0.0
    running_corrects0 = 0
    running_loss1 = 0.0
    running_corrects1 = 0

    for batch_idx, (data, target, cls0_labels, cls1_labels) in enumerate(train_loader):

        data = tuple(d.to(device) for d in data)
        target = target.to(device)
        cls0_labels = cls0_labels.to(device)
        cls0_labels = cls0_labels.long()
        cls1_labels = cls1_labels.to(device)
        cls1_labels = cls1_labels.long()

        # (1) Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
        model.eval()
        model_dis.train()
        optimizer_dis.zero_grad()

        ft0_outputs, cls0_outputs = model(data[0])
        dis0_outputs = model_dis(ft0_outputs.detach()).view(-1)
        label0 = torch.full((data[0].size(0),), A_label, device=device)
        errD0 = criterion_dis(dis0_outputs, label0)

        ft1_outputs, cls1_outputs = model(data[1])
        dis1_outputs = model_dis(ft1_outputs.detach()).view(-1)
        label1 = torch.full((data[1].size(0),), B_label, device=device)
        errD1 = criterion_dis(dis1_outputs, label1)

        errD = errD0 + errD1
        
        # loss_Ldis
        loss_Ldis = np.append(loss_Ldis,errD.cpu().detach().numpy())
        
        errD.backward()

        optimizer_dis.step()

        # (2) Update G network
        
        model_dis.eval()
        model.train()
        optimizer_ft.zero_grad()
        set_parameter_requires_grad(model)

        # i
        dis0_outputs = model_dis(ft0_outputs).view(-1)
        label0 = torch.full((data[0].size(0),), B_label, device=device)
        errG_dis0 = criterion_dis(dis0_outputs, label0)

        dis1_outputs = model_dis(ft1_outputs).view(-1)
        label1 = torch.full((data[1].size(0),), A_label, device=device)
        errG_dis1 = criterion_dis(dis1_outputs, label1)

        #ii
        ft_outputs = (ft0_outputs, ft1_outputs)
        ft_loss_inputs = ft_outputs
        if target is not None:
            target = (target,)
            ft_loss_inputs += target
        ft_loss_outputs = criterion_sia(*ft_loss_inputs)
        ft_loss = ft_loss_outputs[0] if type(ft_loss_outputs) in (tuple, list) else ft_loss_outputs

        #iii
        errG_cls0 = criterion_ft(cls0_outputs, cls0_labels)
        errG_cls1 = criterion_ft(cls1_outputs, cls1_labels)
        _, preds0 = torch.max(cls0_outputs, 1)
        _, preds1 = torch.max(cls1_outputs, 1)

        errG = lambda_cls * (errG_cls0 + errG_cls1) + lambda_sia * ft_loss + lambda_dm * (errG_dis0 + errG_dis1)

        # loss_L
        loss_L = np.append(loss_L,errG.cpu().detach().numpy())

        errG.backward()
        optimizer_ft.step()

        # statistics
        running_loss0 += errG_cls0.item() * data[0].size(0)
        running_loss1 += errG_cls1.item() * data[1].size(0)
        running_corrects0 += torch.sum(preds0 == cls0_labels.data)
        running_corrects1 += torch.sum(preds1 == cls1_labels.data)

    epoch_loss0 = running_loss0 / len(train_loader.dataset)
    epoch_acc0 = running_corrects0.double() / len(train_loader.dataset)
    print('{} Loss0: {:.4f} Acc: {:.4f}'.format('train', epoch_loss0, epoch_acc0))
    epoch_loss1 = running_loss1 / len(train_loader.dataset)
    epoch_acc1 = running_corrects1.double() / len(train_loader.dataset)
    print('{} Loss1: {:.4f} Acc: {:.4f}'.format('train', epoch_loss1, epoch_acc1))

    return loss_L, loss_Ldis

##########################################################################################
# Set Model Parameters requires_grad attribute
def set_parameter_requires_grad(model):
    for name,param in model.named_parameters():
        param.requires_grad = True
        # if 'layer4' in name:
        #     param.requires_grad = True
        # elif 'fc' in name:
        #     param.requires_grad = True
        # else:
        #     param.requires_grad = False

##########################################################################################
# Test stage
def adv_test_epoch(test_loader, model, device):

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    epoch_acc = adv_acc_compute(model, device, test_loader)

    return epoch_acc

##########################################################################################
# save model
def save_model(epoch, saving_model_name, model):
    s_name = saving_model_name + str(epoch+1) + '.pth'
    torch.save(model.state_dict(), s_name)