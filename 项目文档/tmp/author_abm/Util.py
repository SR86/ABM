from __future__ import print_function
import os

import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
from torch.autograd import Variable
import kornia.augmentation as A
import random

#################WaNet#######################

class PostTensorTransform(torch.nn.Module):
    def __init__(self):
        input_height = 32
        input_width = 32
        random_crop = 5
        random_rotation = 10
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((input_height, input_width), padding=random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(random_rotation), p=0.5)
        # 只有cifar10
        self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


def train_Model_with_suppress_wanet(optimizer, model, dataloader, alpha):
    model.train()
    training_loss = 0
    criterionCls = nn.CrossEntropyLoss()
    for batch_idx, (data, target, ispoision, data2, _) in enumerate(dataloader):
        data, target, data2 = Variable(data).float().cuda(), Variable(target).cuda(), Variable(data2).float().cuda()
        bs, _, _, _ = data.shape

        # Prepare grid
        input_height = 32
        k = 4
        s = 0.5
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
            .to("cuda")
        )
        grid_rescale = 1
        array1d = torch.linspace(-1, 1, steps=input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...].to("cuda")
        target_label = 0

        # 0.1投毒率 0.2定死
        transforms = PostTensorTransform().to("cuda")
        num_bd = int(bs * 0.1)
        num_cross = int(num_bd * 0.2)
        grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)
        ins = torch.rand(num_cross, input_height, input_height, 2).to("cuda") * 2 - 1
        grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        inputs_bd = F.grid_sample(data[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        targets_bd = torch.ones_like(target[:num_bd]) * target_label
        inputs_cross = F.grid_sample(data[num_bd: (num_bd + num_cross)], grid_temps2, align_corners=True)
        total_inputs = torch.cat([inputs_bd, inputs_cross, data[(num_bd + num_cross):]], dim=0)
        total_inputs = transforms(total_inputs)
        total_targets = torch.cat([targets_bd, target[num_bd:]], dim=0)

        inputs_bd2 = F.grid_sample(data2[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        inputs_cross2 = F.grid_sample(data2[num_bd: (num_bd + num_cross)], grid_temps2, align_corners=True)
        total_inputs2 = torch.cat([inputs_bd2, inputs_cross2, data2[(num_bd + num_cross):]], dim=0)
        total_inputs2 = transforms(total_inputs2)

        optimizer.zero_grad()
        _, _, near, output = model(total_inputs)
        _, _, near2, _ = model(total_inputs2)
        loss = criterionCls(output, total_targets) \
               + torch.sum((near - near2) * (near - near2)) / 2000 * alpha
        loss.backward()
        optimizer.step()
        training_loss += loss

    return training_loss / len(dataloader.dataset)


def test_Model_wanet(model, dataloader):
    model.eval()
    correct = 0
    activation = 0
    for data, target, _, _, _, in dataloader:
        with torch.no_grad():
            data, target = Variable(data).float().cuda(), Variable(target).cuda()
            bs, _, _, _ = data.shape

            input_height = 32
            k = 4
            s = 0.5
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            noise_grid = (
                F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
                .to("cuda")
            )
            grid_rescale = 1
            array1d = torch.linspace(-1, 1, steps=input_height)
            x, y = torch.meshgrid(array1d, array1d)
            identity_grid = torch.stack((y, x), 2)[None, ...].to("cuda")
            target_label = 0

            # Evaluate Backdoor
            grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            inputs_bd = F.grid_sample(data, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            targets_bd = torch.ones_like(target) * target_label
            _, _, _, output = model(inputs_bd)
            for i in range(bs):
                activation = activation + output[i][target[i]]
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets_bd.data.view_as(pred)).cpu().sum()
    activation = activation / len(dataloader.dataset)
    correctRate = 100. * correct / len(dataloader.dataset)

    return activation, correctRate


def train_Model_wanet(optimizer, model, dataloader):
    model.train()
    training_loss = 0
    DateNum = 0
    criterionCls = nn.CrossEntropyLoss()
    activation = 0
    for batch_idx, (_, _, _, data, target) in enumerate(dataloader):
        data, target = Variable(data).float().cuda(), Variable(target).cuda()
        bs, _, _, _ = data.shape

        # Prepare grid
        input_height = 32
        k = 4
        s = 0.5
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
            .to("cuda")
        )
        grid_rescale = 1
        array1d = torch.linspace(-1, 1, steps=input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...].to("cuda")
        target_label = 0

        # 0.1投毒率 0.2定死
        transforms = PostTensorTransform().to("cuda")
        num_bd = int(bs * 0.1)
        num_cross = int(num_bd * 0.2)
        grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)
        ins = torch.rand(num_cross, input_height, input_height, 2).to("cuda") * 2 - 1
        grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        inputs_bd = F.grid_sample(data[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        targets_bd = torch.ones_like(target[:num_bd]) * target_label
        inputs_cross = F.grid_sample(data[num_bd: (num_bd + num_cross)], grid_temps2, align_corners=True)
        total_inputs = torch.cat([inputs_bd, inputs_cross, data[(num_bd + num_cross):]], dim=0)
        total_inputs = transforms(total_inputs)
        total_targets = torch.cat([targets_bd, target[num_bd:]], dim=0)

        optimizer.zero_grad()
        _, _, _, output = model(total_inputs)
        for i in range(bs):
            activation = activation + output[i][total_targets[i]]
        loss = criterionCls(output, total_targets)
        loss.backward()
        optimizer.step()
        training_loss += loss
        DateNum += bs
    return activation / len(dataloader.dataset), training_loss / len(dataloader.dataset)


def test_two_Model_wanet(TeacherModel, StudentModel, dataloader, gamma):
    print("gamma:{}".format(gamma))
    TeacherModel.eval()
    StudentModel.eval()
    correct = 0
    activation = 0
    for data, target, _, _, _, in dataloader:
        with torch.no_grad():
            data, target = Variable(data).float().cuda(), Variable(target).cuda()
            bs, _, _, _ = data.shape

            input_height = 32
            k = 4
            s = 0.5
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            noise_grid = (
                F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
                .to("cuda")
            )
            grid_rescale = 1
            array1d = torch.linspace(-1, 1, steps=input_height)
            x, y = torch.meshgrid(array1d, array1d)
            identity_grid = torch.stack((y, x), 2)[None, ...].to("cuda")
            target_label = 0

            # Evaluate Backdoor
            grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            inputs_bd = F.grid_sample(data, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            targets_bd = torch.ones_like(target) * target_label

            _, _, _, output = TeacherModel(inputs_bd)
            _, _, _, output2 = StudentModel(inputs_bd)
            output = output - output2 * ((torch.sign(output2 - gamma) + 1) / 2) * 2
            for i in range(bs):
                activation = activation + output[i][target[i]]
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets_bd.data.view_as(pred)).cpu().sum()
    activation = activation / len(dataloader.dataset)
    correctRate = 100. * correct / len(dataloader.dataset)

    return activation, correctRate


#################WaNet#######################



def train_Model_with_suppress(optimizer, model, dataloader):
    model.train()
    training_loss = 0
    criterionCls = nn.CrossEntropyLoss()
    for batch_idx, (data, target, ispoision, data2, _) in enumerate(dataloader):
        data, target, data2 = Variable(data).float().cuda(), Variable(target).cuda(), Variable(data2).float().cuda()
        bs, _, _, _ = data.shape
        optimizer.zero_grad()
        _, _, near, output = model(data)
        _, _, near2, _ = model(data2)
        loss = criterionCls(output, target) + torch.sum((near - near2) * (near - near2)) / 2000
        loss.backward()
        optimizer.step()
        training_loss += loss

    return training_loss / len(dataloader.dataset)


def train_Model(optimizer, model, dataloader):
    model.train()
    training_loss = 0
    DateNum = 0
    criterionCls = nn.CrossEntropyLoss()
    activation = 0
    for batch_idx, (_, _, _, data, target) in enumerate(dataloader):
        data, target = Variable(data).float().cuda(), Variable(target).cuda()
        bs, _, _, _ = data.shape
        optimizer.zero_grad()
        _, _, _, output = model(data)
        for i in range(bs):
            activation = activation + output[i][target[i]]
        loss = criterionCls(output, target)
        loss.backward()
        optimizer.step()
        training_loss += loss
        DateNum += bs
    return activation / len(dataloader.dataset), training_loss / len(dataloader.dataset)


def train_Model_adap(optimizer, model, dataloader):
    model.train()
    training_loss = 0
    DateNum = 0
    criterionCls = nn.CrossEntropyLoss()
    activation = 0
    for batch_idx, (data, _, _, _, target) in enumerate(dataloader):
        data, target = Variable(data).float().cuda(), Variable(target).cuda()
        bs, _, _, _ = data.shape
        optimizer.zero_grad()
        _, _, _, output = model(data)
        for i in range(bs):
            activation = activation + output[i][target[i]]
        loss = criterionCls(output, target)
        loss.backward()
        optimizer.step()
        training_loss += loss
        DateNum += bs
    return activation / len(dataloader.dataset), training_loss / len(dataloader.dataset)

def train_Model_distill(optimizer, TeacherModel, StudentModel, dataloader):
    StudentModel.train()
    training_loss = 0
    DateNum = 0
    lossFunction = nn.MSELoss()
    for batch_idx, (data, target, ispoision, _, _) in enumerate(dataloader):
        data, _ = Variable(data).float().cuda(), Variable(target).cuda()
        bs, _, _, _ = data.shape
        optimizer.zero_grad()
        _, _, _, TeacherOutput = TeacherModel(data)
        # TeacherOutput = F.softmax(TeacherOutput, dim=1)
        _, _, _, StudentOutput = StudentModel(data)
        # StudentOutput = F.softmax(StudentOutput, dim=1)
        # TeacherPred = TeacherOutput.data.max(1, keepdim=True)[1]
        # StudentOutputTemp=StudentOutput.clone()
        # TeacherOutputTemp=TeacherOutput.clone()

        # for i in range(bs):
        #     StudentOutput[i][target[i]]=-1000
        #     TeacherOutput[i][target[i]]=-1000

        # print(StudentOutputTemp[0])
        # print(TeacherOutputTemp[0])
        # print(target[0])
        # print(ispoision[0])

        loss = lossFunction(StudentOutput, TeacherOutput)
        loss.backward()
        optimizer.step()
        training_loss += loss
        DateNum += bs
    return DateNum, training_loss / len(dataloader.dataset)

def train_Model_distill_one(opt, optimizer, TeacherModel, StudentModel, dataloader):
    StudentModel.train()
    training_loss = 0
    DateNum = 0
    lossFunction = nn.MSELoss()
    for batch_idx, (data, target, ispoision, _, _) in enumerate(dataloader):
        data, _ = Variable(data).float().cuda(), Variable(target).cuda()
        bs, _, _, _ = data.shape
        optimizer.zero_grad()
        _, _, _, TeacherOutput = TeacherModel(data)
        _, _, _, StudentOutput = StudentModel(data)
        # TeacherOutputTemp = torch.zeros([bs, 1]).cuda()
        # for i in range(bs):
        #     TeacherOutputTemp[i][opt.target_label] = TeacherOutput[i][opt.target_label]
        # loss = lossFunction(StudentOutput, TeacherOutputTemp)
        loss = lossFunction(StudentOutput, TeacherOutput)
        loss.backward()
        optimizer.step()
        training_loss += loss
        DateNum += bs
    return DateNum, training_loss / len(dataloader.dataset)

def test_two_Model(TeacherModel, StudentModel, dataloader, gamma):
    print("gamma:{}".format(gamma))
    TeacherModel.eval()
    StudentModel.eval()
    correct = 0
    activation = 0
    target_label = 0
    for data, target, _, _, _, in dataloader:
        with torch.no_grad():
            data, target = Variable(data).float().cuda(), Variable(target).cuda()
            bs, _, _, _ = data.shape
            _, _, _, output = TeacherModel(data)
            _, _, _, output2 = StudentModel(data)
            output = output - output2 * ((torch.sign(output2 - gamma) + 1) / 2) * 2
            for i in range(bs):
                activation = activation + output[i][target[i]]
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    activation = activation / len(dataloader.dataset)
    correctRate = 100. * correct / len(dataloader.dataset)

    return activation, correctRate

def test_Model(model, dataloader):
    model.eval()
    correct = 0
    activation = 0
    for data, target, _, _, _, in dataloader:
        with torch.no_grad():
            data, target = Variable(data).float().cuda(), Variable(target).cuda()
            bs, _, _, _ = data.shape
            _, _, _, output = model(data)
            for i in range(bs):
                activation = activation + output[i][target[i]]
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    activation = activation / len(dataloader.dataset)
    correctRate = 100. * correct / len(dataloader.dataset)

    return activation, correctRate

def test_Model_each_class(model, dataloader):
    model.eval()
    sum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    activate = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for data, target, ispoision, _, _, in dataloader:
        with torch.no_grad():
            data, target = Variable(data).float().cuda(), Variable(target).cuda()
            bs, _, _, _ = data.shape
            _, _, _, output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            for i in range(bs):
                sum[target[i]]+=1
                if target[i]==pred[i]:
                    correct[target[i]]+=1
                activate[target[i]]+=output[i][target[i]].item()

    print(sum)
    print(correct)
    print(activate)

    return sum, correct

def test_Model_With_Stu_Tea(StudentModel, TeacherModel, dataloader):
    StudentModel.eval()
    TeacherModel.eval()
    correct = 0
    activation = 0
    for data, target, _, _, _, in dataloader:
        with torch.no_grad():
            data, target = Variable(data).float().cuda(), Variable(target).cuda()
            bs, _, _, _ = data.shape
            _, _, _, TeacherOutput = TeacherModel(data)
            _, _, _, StudentOutput = StudentModel(data)
            for i in range(bs):
                activation = activation + TeacherOutput[i][target[i]]
                TeacherOutput[i][0] = TeacherOutput[i][0] - StudentOutput[i][0]
            pred = TeacherOutput.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    activation = activation / len(dataloader.dataset)
    correctRate = 100. * correct / len(dataloader.dataset)

    return activation, correctRate


def adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 2:
        lr = lr
    elif epoch < 20:
        lr = 0.01
    elif epoch < 30:
        lr = 0.0001
    else:
        lr = 0.0001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
