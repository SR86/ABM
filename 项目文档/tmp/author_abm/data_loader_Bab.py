import copy
from PIL import Image
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import numpy as np
import time
from tqdm import tqdm
import torch.nn.functional as F
from numba import jit
# from numba.types import float64, int64

def get_train_loader(opt):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.ToTensor(),
    ])

    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root=opt.datasetPath, train=True, download=True)
    else:
        raise Exception('Invalid dataset')
    train_data = DatasetCL(opt, full_dataset=trainset, transform=tf_train)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    return train_loader


def get_test_loader(opt, trigger_type, target_label):
    print('==> Preparing test data..')
    tf_test = transforms.Compose([transforms.ToTensor()])
    if (opt.dataset == 'CIFAR10'):
        testset = datasets.CIFAR10(root=opt.datasetPath, train=False, download=True)
    else:
        raise Exception('Invalid dataset')

    test_data_clean = DatasetBD(opt, trigger_type=trigger_type, target_label=target_label, full_dataset=testset,
                                inject_portion=0, transform=tf_test, mode='test')
    test_data_bad = DatasetBD(opt, trigger_type=trigger_type, target_label=target_label, full_dataset=testset,
                              inject_portion=1, transform=tf_test, mode='test')

    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean, batch_size=opt.batch_size, shuffle=False, )
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad, batch_size=opt.batch_size, shuffle=False, )
    return test_clean_loader, test_bad_loader


def get_backdoor_loader(opt, inject_portion, trigger_type, target_label):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([transforms.ToTensor()])
    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root=opt.datasetPath, train=True, download=False)
    else:
        raise Exception('Invalid dataset')

    train_data_bad = DatasetBD(opt, trigger_type=trigger_type, target_label=target_label, full_dataset=trainset,
                               inject_portion=inject_portion, transform=tf_train, mode='train')
    train_bad_loader = DataLoader(dataset=train_data_bad, batch_size=opt.batch_size, shuffle=False, )
    return train_data_bad, train_bad_loader


class Dataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        ispoision = self.dataset[index][2]

        if self.transform:
            image = self.transform(image)
        return image, label, ispoision, image, label

    def __len__(self):
        return self.dataLen


class DatasetCL(Dataset):
    def __init__(self, opt, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=opt.ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))
        return train_dataset


class DatasetBD(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, trigger_type, target_label, transform=None, mode="train",
                 device=torch.device("cuda"), ):
        self.opt = opt
        self.clamp_batch_pert = torch.load('./data/cifar-10-noisy.pt')
        self.dataset = self.addTrigger(full_dataset, target_label, inject_portion, mode,
                                       trigger_type, opt.target_type, opt.inject_portion_defend,
                                       opt.trigger_type_defend, opt.target_label_defend)
        self.device = device
        self.transform = transform


    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)
        ispoision = self.dataset[item][2]
        img2 = self.dataset[item][3]
        orilabel = self.dataset[item][4]

        return img, label, ispoision, self.transform(img2), orilabel

    def __len__(self):
        return len(self.dataset)


    def addTrigger(self, dataset, target_label, inject_portion, mode, trigger_type,
                   target_type, inject_portion_defend, trigger_type_defend, target_label_defend):
        print("Generating " + mode + "bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]

        perm2 = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion_defend)]
        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            ##用于CBA
            img2 = np.array(dataset[(i + 1) % 30][0])

            if target_type == 'all2one':
                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if trigger_type == "RefoolTrigger" or trigger_type == 'NarTrigger':
                        if data[1] != target_label:
                            if i in perm2:
                                # defend data
                                temp = copy.deepcopy(img)
                                img = self.selectTrigger(img, img2, width, height, trigger_type_defend,mode=mode)
                                dataset_.append((img, target_label_defend, 2, temp, data[1]))
                            else:
                                # clean data
                                dataset_.append((img, data[1], 0, img, data[1]))
                        elif cnt < int(len(dataset) * inject_portion):
                            # select trigger
                            img = self.selectTrigger(img, img2, width, height, trigger_type,mode=mode)
                            # poision data
                            dataset_.append((img, target_label, 1, img, target_label))
                            cnt += 1

                    else:
                        if i in perm:
                            # select trigger
                            img = self.selectTrigger(img, img2, width, height, trigger_type,mode=mode)
                            # poision data
                            dataset_.append((img, target_label, 1, img, target_label))
                            cnt += 1
                        else:
                            if i in perm2:
                                # defend data
                                temp = copy.deepcopy(img)
                                img = self.selectTrigger(img, img2, width, height, trigger_type_defend,mode=mode)
                                dataset_.append((img, target_label_defend, 2, temp, data[1]))
                            else:
                                # clean data
                                dataset_.append((img, data[1], 0, img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, img2, width, height, trigger_type,mode=mode)

                        dataset_.append((img, target_label, 1, img, data[1]))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1], 0, img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")
        return dataset_

    def selectTrigger(self, img, img2, width, height, triggerType,mode):

        assert triggerType in ['SIGTrigger', 'BadNetTrigger', 'BlendTrigger', 'CBATrigger', 'RefoolTrigger',
                               'AnotherBadNetTrigger','BppTrigger','NarTrigger','WaNetTrigger']

        if triggerType == 'SIGTrigger':
            img = self._SIGTrigger(img, img2, width, height)
        elif triggerType == 'BadNetTrigger':
            img = self._BadNetTrigger(img, img2, width, height)
        elif triggerType == 'BlendTrigger':
            img = self._BlendTrigger(img, img2, width, height)
        elif triggerType == 'CBATrigger':
            img = self._CBATrigger(img, img2, width, height)
        elif triggerType == 'RefoolTrigger':
            img = self._RefoolTrigger(img, img2, width, height)
        elif triggerType == 'AnotherBadNetTrigger':
            img = self._AnotherBadNetTrigger(img, img2, width, height)
        elif triggerType == 'BppTrigger':
            img = np.transpose((img), (2, 0, 1))
            img = self.BPP(img, 8)
            img = np.transpose((img), (1, 2, 0))
        elif triggerType == 'NarTrigger':
            img = self._NarTrigger(img, img2, width, height,mode)
        elif triggerType == 'WaNetTrigger':
            None
        else:
            raise NotImplementedError

        return img

    def _SIGTrigger(self, img, img2, width, height):
        alpha = 0.2
        # load signal mask
        signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)
        # plt.imshow(img, cmap='gray')
        # plt.axis('off')
        # plt.show()
        return blend_img

    def _BadNetTrigger(self, img, img2, width, height):
        for c in range(3):
            img[27, 27, c] = 255
            img[28, 28, c] = 255
            img[27, 28, c] = 255
            img[28, 27, c] = 255

        # plt.imshow(img, cmap='gray')
        # plt.axis('off')
        # plt.show()
        return img

    def _AnotherBadNetTrigger(self, img, img2, width, height):
        for c in range(3):
            img[0, 0, c] = 255
            img[1, 1, c] = 255
            img[1, 0, c] = 255
            img[0, 1, c] = 255
        # plt.imshow(img, cmap='gray')
        # plt.axis('off')
        # plt.show()
        return img

    def _BlendTrigger(self, img, img2, width, height):
        alpha = 0.2
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)
        return blend_img
        # style = np.array(Image.open('data/img_1.png'))
        # img = img * 0.95 + style * 0.05
        # plt.imshow(img/255, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # return img

    def _CBATrigger(self, img, img2, width, height):
        img = np.transpose(self.composite(img, img2), (1, 2, 0))
        # plt.imshow(img, cmap='gray')
        # plt.axis('off')
        # plt.show()
        return img

    def composite(self, data1, data2):
        data1 = torch.tensor(np.transpose(data1, (2, 0, 1)))
        data2 = torch.tensor(np.transpose(data2, (2, 0, 1)))
        data2 = data2.reshape([1, 3, 32, 32])
        data2 = F.interpolate(data2, 16)
        data2 = data2.reshape([3, 16, 16])
        # temp = random.randint(0, 15)
        temp = 0

        for c in range(3):
            for i in range(16):
                for j in range(16):
                    data1[c][i + temp + 10][j] = data2[c][i][j]

        return data1.numpy()

    def _RefoolTrigger(self, img, img2, width, height):
        style = np.array(Image.open('data/img_1.png'))
        img = self.Refool_image(np.transpose(img, (1, 2, 0)), np.transpose(style, (1, 2, 0)))
        # plt.imshow(img, cmap='gray')
        # plt.axis('off')
        # plt.show()
        return img

    def Refool_image(self, img_t, img_r):
        img_t = img_t
        img_r = img_r
        weight_t = np.mean(img_t)
        weight_r = np.mean(img_r)

        param_t = weight_t / (weight_t + weight_r)
        param_r = weight_r / (weight_t + weight_r)
        img_b = np.uint8(np.clip(param_t * img_t / 255. + param_r * img_r / 255., 0, 1) * 255)
        return np.transpose((img_b), (2, 0, 1))

    # @jit(nopython=True)
    def BPP(self,image, squeeze_num):
        channel, h, w = image.shape
        for y in range(h):
            for x in range(w):
                old = image[:, y, x]
                temp = np.empty_like(old).astype(np.float64)
                new = np.round_(old / 255.0 * (squeeze_num - 1), 0, temp) / (squeeze_num - 1) * 255
                error = old - new
                image[:, y, x] = new
                if x + 1 < w:
                    image[:,y, x + 1] = image[:,y, x + 1]+error * 0.4375
                if (y + 1 < h) and (x + 1 < w):
                    image[:, y + 1, x + 1] =image[:, y + 1, x + 1] +error * 0.0625
                if y + 1 < h:
                    image[:, y + 1, x] =image[:, y + 1, x] +error * 0.3125
                if (x - 1 >= 0) and (y + 1 < h):
                    image[:, y + 1, x - 1] =image[:, y + 1, x - 1]+error * 0.1875
        return image

    def _NarTrigger(self, img, img2, width, height,mode):
        i=img
        tsf_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img1=tsf_train(img)
        if mode=='test':
            img = torch.clamp(self.apply_noise_patch(self.clamp_batch_pert*3, img1.clone(), mode='add'), -1, 1)
        else:
            img = torch.clamp(self.apply_noise_patch(self.clamp_batch_pert, img1.clone(), mode='add'), -1, 1)
        denormalized_image = img * 0.5 + 0.5
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(denormalized_image)
        np_image = np.array(pil_image)
        return np_image


    def apply_noise_patch(self,noise, images, offset_x=0, offset_y=0, mode='change', padding=20, position='fixed'):
        '''
        noise: torch.Tensor(1, 3, pat_size, pat_size)
        images: torch.Tensor(N, 3, 512, 512)
        outputs: torch.Tensor(N, 3, 512, 512)
        '''
        length = images.shape[2] - noise.shape[2]
        if position == 'fixed':
            wl = offset_x
            ht = offset_y
        else:
            wl = np.random.randint(padding, length - padding)
            ht = np.random.randint(padding, length - padding)
        if images.dim() == 3:
            noise_now = noise.clone()[0, :, :, :]
            wr = length - wl
            hb = length - ht
            m = nn.ZeroPad2d((wl, wr, ht, hb))
            if (mode == 'change'):
                images[:, ht:ht + noise.shape[2], wl:wl + noise.shape[3]] = 0
                images += m(noise_now)
            else:
                images += noise_now
        else:
            for i in range(images.shape[0]):
                noise_now = noise.clone()
                wr = length - wl
                hb = length - ht
                m = nn.ZeroPad2d((wl, wr, ht, hb))
                if (mode == 'change'):
                    images[i:i + 1, :, ht:ht + noise.shape[2], wl:wl + noise.shape[3]] = 0
                    images[i:i + 1] += m(noise_now)
                else:
                    images[i:i + 1] += noise_now
        return images