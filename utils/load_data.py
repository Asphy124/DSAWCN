import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets



def load_data(data_name=None, gcn=False, split_data=0.2, batch_size=16):    
    if data_name == 'dtd':
        if gcn:
            normalize = GCN()
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose(
                            [transforms.Resize(256),
                             transforms.RandomCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip(),
                             transforms.ToTensor(),
                             normalize
                             ])
        transform_test = transforms.Compose(
                            [transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize
                            ])
        
        training_data = datasets.dtd.DTD(root='./data/', split="train", partition=10,
                                         download=True, transform=transform_train)
        val_data = datasets.dtd.DTD(root='./data/', split="val", partition=10,
                                           download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
        num_classes = 47

        
    elif data_name == 'kth':
        if gcn:
            normalize = GCN()
        else:
            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose(
                            [transforms.Resize((256, 256)),
                             transforms.RandomCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip(),
                             transforms.ToTensor(),
                             normalize
                             ])
        transform_test = transforms.Compose(
                            [transforms.Resize((256, 256)),
                             transforms.CenterCrop((224, 224)),
                             transforms.ToTensor(),
                             normalize
                             ])

        kth_train_dataset = datasets.ImageFolder(root='./data/kth/train', transform=transform_train)
        kth_test_dataset = datasets.ImageFolder(root='./data/kth/test', transform=transform_test)
        train_loader = torch.utils.data.DataLoader(kth_train_dataset, shuffle=True, batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(kth_test_dataset, shuffle=True, batch_size=batch_size)
        num_classes = 11

    elif data_name == 'cifar-10':
        if gcn:
            normalize = GCN()
        else:
            normalize = transforms.Normalize(mean=[0.4914, 0.4821, 0.4465],
                                            std=[0.2470, 0.2435, 0.2616])

        transform_train = transforms.Compose(
                            [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize
                            ])
        transform_test = transforms.Compose(
                            [transforms.ToTensor(),
                            normalize
                            ])
        
        train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10("./data/cifar10", train=True, download=True, transform=transform_train),
                batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10("./data/cifar10", train=False, download=True, transform=transform_test),
                batch_size=batch_size, shuffle=True) 
        num_classes = 10 

    else:
        if gcn:
            normalize = GCN()
        else:
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5]) # 要计算
        transform_train = transforms.Compose([
                             transforms.Resize(300),
                             transforms.RandomCrop(256),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip(),
                             transforms.ToTensor(),
                             normalize
                             ])
        transform_test = transforms.Compose([
                             transforms.Resize(300),
                             transforms.CenterCrop(256),
                             transforms.ToTensor(),
                             normalize
                             ])
        if data_name == 'barkvn-50':
            train_set = datasets.ImageFolder(
                root="./data/BarkVN-50", transform=transform_train)
            val_set = datasets.ImageFolder(
                root="./data/BarkVN-50", transform=transform_test)
            num_classes = 50
        elif data_name == 'trunk12':
            train_set = datasets.ImageFolder(
                root="./data/Trunk12", transform=transform_train)
            val_set = datasets.ImageFolder(
                root="./data/Trunk12", transform=transform_test)
            num_classes = 12
        elif data_name == 'bark-20':
            train_set = datasets.ImageFolder(
                root="./data/barknet20", transform=transform_train)
            val_set = datasets.ImageFolder(
                root="./data/barknet20", transform=transform_test)
            num_classes = 20
        elif data_name == 'bark-54':
            train_set = datasets.ImageFolder(
                root="./data/barktexture54", transform=transform_train)
            val_set = datasets.ImageFolder(
                root="./data/barktexture54", transform=transform_test)
            num_classes = 54
        else:
            raise "Unknown dataset"
        
        num_train = len(train_set)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(split_data * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                    sampler=train_sampler, num_workers=0)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                    sampler=val_sampler, num_workers=0)
    return train_loader, val_loader, num_classes
    

class GCN(object):
    """
    全局对比度归一化
    """
    def __init__(self,
                 channel_wise=True,
                 scale=1.,
                 sqrt_bias=10.,
                 epsilon=1e-8):
        
        self.scale = scale
        self.sqrt_bias = sqrt_bias
        self.channel_wise = channel_wise
        self.epsilon = epsilon

    def __call__(self, img):
        if self.channel_wise:  # 如果是RGB图像
            assert(img.shape[0] == 3)  # assert断言，如果表达式为false(不是RGB图)，则触发异常
            for i in range(3):
                img[i, :, :] = img[i, :, :] - torch.mean(img[i, :, :])
                norm = max(self.epsilon, torch.sqrt(self.sqrt_bias + torch.var(img[i, :, :])))
                img[i, :, :] = img[i, :, :] / norm
            img = img * self.scale
            return img
        else:
            img = img - torch.mean(img)
            norm = max(self.epsilon, torch.sqrt(self.sqrt_bias + torch.var(img)))
            img = img * self.scale / norm
            return img

    def __repr__(self):
        return self.__class__.__name__
    

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

