import torch 
import torch.nn as nn
import numpy as np
import pywt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_cmf(filter, kernel_size):
    if filter.shape[-1] > 1:
        cm_operator = torch.tensor([(-1) ** i for i in range(kernel_size)]).reshape((1, 1, 1, -1)).to("cuda")
        cm_filter = torch.flip(filter, [3]).to("cuda") * cm_operator
    else:
        cm_operator = torch.tensor([(-1) ** i for i in range(kernel_size)]).reshape((1, 1, -1, 1)).to("cuda")
        cm_filter = torch.flip(filter, [2]).to("cuda") * cm_operator
    return cm_filter


class downsample_conv_block(nn.Module):
    def __init__(self, level, in_plane):
        super(downsample_conv_block, self).__init__()
        blks = []
        
        for j in range(level):
            blks_tmp = []
            blks1 = nn.Sequential(
                nn.Conv2d(3*in_plane, in_plane, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(in_plane),
                nn.ReLU(inplace=True)
            )
            blks_tmp.append(blks1)
            for _ in range(level - j - 1):
                blks2 = nn.Sequential(
                    nn.Conv2d(in_plane, in_plane, kernel_size=3, stride=2, padding=1, bias=False, groups=in_plane),
                    nn.BatchNorm2d(in_plane),
                    nn.ReLU(inplace=True)
                )
                blks_tmp.append(blks2)
            blks_tmp = nn.Sequential(*blks_tmp).to(device)
            blks.append(blks_tmp)
            self.blks = nn.ModuleList([*blks])


class WN_SubBlock(nn.Module):  # 一个Block，包含一个低通滤波器和一个高通滤波器
    def __init__(self,
                 horizontal=True,
                 filter_Init=0,
                 kernel_size = 8,
                 trainable=[True, True],
                 in_channels=3,
                 ):
        super(WN_SubBlock, self).__init__()
        self.horizontal = horizontal
        self.in_channels = in_channels
        self.trainable = trainable
        self.kernel_size = kernel_size
        self.pad_size = kernel_size - 2
        if horizontal:
            s = (1, 2)
            filter_Init = filter_Init.permute(0, 1, 3, 2)
        else:
            s = (2, 1)

        low_pass_filter = torch.cat([filter_Init] * in_channels, dim=0)
        high_pass_filter = get_cmf(low_pass_filter, kernel_size)  # 高通滤波器

        w1 = nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size, stride=s, bias=False, groups=in_channels)  # channels需要修改
        w1.weight = nn.Parameter(data=low_pass_filter.clone(), requires_grad=trainable[0])  # 初始化完成
        self.kernel_Low = nn.Sequential(w1, nn.Tanh())
        # self.kernel_Low = nn.Sequential(w1, )

        w2 = nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size, stride=s, bias=False, groups=in_channels)
        w2.weight = nn.Parameter(data=high_pass_filter.clone(), requires_grad=trainable[1])
        self.kernel_High = nn.Sequential(w2, nn.Tanh())
        # self.kernel_High = nn.Sequential(w2, )


    def forward(self, input_data):
        input_data = pad(input_data, self.pad_size, horizontal=self.horizontal)
        low_coeff = self.kernel_Low(input_data)
        if not self.trainable[1]:
            self.kernel_High[0].weight = nn.Parameter(data=get_cmf(self.kernel_Low[0].weight, self.kernel_size).clone(), requires_grad=False)
        high_coeff = self.kernel_High(input_data)
        return low_coeff, high_coeff



class WNBlock(nn.Module):  # 一个Block，包含三个WN_SubBlock
    def __init__(self, wavelet="db4", kernel_size=8, in_channels=3, mode="Stable") -> None:
        super(WNBlock, self).__init__()
        self.mode = mode

        filter_Init = pywt.Wavelet(wavelet).filter_bank[2]
        filter_Init = np.array(filter_Init)[np.newaxis, np.newaxis, :, np.newaxis].copy()  # 浅复制
        # 转为float32
        filter_Init = torch.from_numpy(filter_Init).float().to("cuda")

        if (mode == "Free") or (mode == "CQF_All_All"):  # 块内全训练
            self.wn1 = WN_SubBlock(horizontal=True, 
                    filter_Init=filter_Init, 
                    kernel_size=kernel_size, 
                    trainable=[True,True], 
                    in_channels=in_channels)
            self.wn2 = WN_SubBlock(horizontal=False, 
                    filter_Init=filter_Init, 
                    kernel_size=kernel_size, 
                    trainable=[True,True], 
                    in_channels=in_channels)
            self.wn3 = WN_SubBlock(horizontal=False, 
                    filter_Init=filter_Init, 
                    kernel_size=kernel_size, 
                    trainable=[True,True], 
                    in_channels=in_channels)

        if mode == "Stable":  # 块内全不训练
            self.wn1 = WN_SubBlock(horizontal=True, 
                                filter_Init=filter_Init, 
                                kernel_size=kernel_size, 
                                trainable=[False,False], 
                                in_channels=in_channels)
            self.wn2 = WN_SubBlock(horizontal=False,
                                filter_Init=filter_Init,
                                kernel_size=kernel_size,
                                trainable=[False,False],
                                in_channels=in_channels)
            
            self.wn3 = WN_SubBlock(horizontal=False,
                                filter_Init=filter_Init,
                                kernel_size=kernel_size,
                                trainable=[False,False],
                                in_channels=in_channels)
        
        elif (mode == "CQF_Low_1") or (mode == "Layer_Low_1"):  # 块内只训练第一个低频滤波器,CQF辐射到全局,Layer每层均可训练
            self.wn1 = WN_SubBlock(horizontal=True, 
                                filter_Init=filter_Init, 
                                kernel_size=kernel_size, 
                                trainable=[True,False], 
                                in_channels=in_channels)
            self.wn2 = WN_SubBlock(horizontal=False,
                                filter_Init=filter_Init,
                                kernel_size=kernel_size,
                                trainable=[False,False],
                                in_channels=in_channels)
            
            self.wn3 = WN_SubBlock(horizontal=False,
                                filter_Init=filter_Init,
                                kernel_size=kernel_size,
                                trainable=[False,False],
                                in_channels=in_channels)
        
        elif (mode == "CQF_All_1_Filter") or (mode == "Layer_All_1_Filter"):  # 块内只训练第一个低频滤波器和第一个高频滤波器，辐射到全局
            self.wn1 = WN_SubBlock(horizontal=True, 
                                filter_Init=filter_Init, 
                                kernel_size=kernel_size, 
                                trainable=[True,True], 
                                in_channels=in_channels)
            self.wn2 = WN_SubBlock(horizontal=False,
                                filter_Init=filter_Init,
                                kernel_size=kernel_size,
                                trainable=[False,False],
                                in_channels=in_channels)
            
            self.wn3 = WN_SubBlock(horizontal=False,
                                filter_Init=filter_Init,
                                kernel_size=kernel_size,
                                trainable=[False,False],
                                in_channels=in_channels)
        
        elif (mode == "CQF_Low_All") or (mode == "Layer_Low_All"):  # 块内训练每个子块的低通滤波器，辐射到全局
            self.wn1 = WN_SubBlock(horizontal=True, 
                                filter_Init=filter_Init, 
                                kernel_size=kernel_size, 
                                trainable=[True,False], 
                                in_channels=in_channels)
            self.wn2 = WN_SubBlock(horizontal=False,
                                filter_Init=filter_Init,
                                kernel_size=kernel_size,
                                trainable=[True,False],
                                in_channels=in_channels)
            self.wn3 = WN_SubBlock(horizontal=False,
                                filter_Init=filter_Init,
                                kernel_size=kernel_size,
                                trainable=[True,False],
                                in_channels=in_channels)


        
    def forward(self, input_data):
        low_coeff, high_coeff = self.wn1(input_data)
        if (self.mode == "CQF_Low_1") or (self.mode == "Layer_Low_1") or (self.mode == "CQF_All_1_Filter") or (self.mode == "Layer_All_1_Filter"):
            tmplow = torch.nn.Parameter(self.wn1.kernel_Low[0].weight.data.permute(0, 1, 3, 2), requires_grad=False)
            tmphigh = torch.nn.Parameter(self.wn1.kernel_High[0].weight.data.permute(0, 1, 3, 2), requires_grad=False)
            self.wn2.kernel_Low[0].weight = tmplow 
            self.wn3.kernel_Low[0].weight = tmplow 
            self.wn2.kernel_High[0].weight = tmphigh 
            self.wn3.kernel_High[0].weight = tmphigh 
        
        ll, lh = self.wn2(low_coeff)
        hl, hh = self.wn3(high_coeff)

        return (low_coeff, high_coeff, ll, lh, hl, hh) 
    


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # This disable the conv if compression rate is equal to 1
        self.disable_conv = in_planes == out_planes
        if not self.disable_conv:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)

    def forward(self, x):
        if self.disable_conv:
            return self.relu(self.bn1(x))
        else:
            return self.conv1(self.relu(self.bn1(x)))


class LevelWNBlocks(nn.Module):
    def __init__(self, 
                 wavelet,
                 in_channel,
                 kernel_size,
                 regu_details,
                 regu_approx,
                 bottleneck=False,
                 mode="Stable"):
        super(LevelWNBlocks, self).__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        self.bottleneck = bottleneck
        self.loss_details = nn.SmoothL1Loss()
        self.waveletst = WNBlock(wavelet,
                               kernel_size,
                               in_channel,
                               mode=mode)
        if bottleneck:
            self.bottleneck = BottleneckBlock(in_channel, in_channel)
    
    def forward(self, x):
        (c, d, ll, lh, hl, hh) = self.waveletst(x)  #低频和高频
        details=torch.cat([lh, hl, hh],1)
        r = None
        if(self.regu_approx + self.regu_details != 0.0):
            if self.regu_details:
                rd = self.regu_details * self.loss_details(d, torch.zeros(d.size()).cuda())
                rd += self.regu_details * self.loss_details(lh, torch.zeros(lh.size()).cuda())
                rd += self.regu_details * self.loss_details(hh, torch.zeros(hh.size()).cuda())
            if self.regu_approx:
                rc = self.regu_approx * torch.dist(c.mean(), ll.mean(), p=2)
                rc += self.regu_approx * torch.dist(ll.mean(), c.mean(), p=2)
                rc += self.regu_approx * torch.dist(hl.mean(), d.mean(), p=2)

            if self.regu_approx == 0.0:
                r = rd
            elif self.regu_details == 0.0:
                r = rc
            else:
                r = rd + rc
        if self.bottleneck:
            ll = self.bottleneck(ll)
        return ll, r, details
    
    def inverse(self, ll, details, osig):
        lh = details[:, 0:ll.size()[1], :, :]
        hl = details[:, ll.size()[1]:2*ll.size()[1], :, :]
        hh = details[:, 2*ll.size()[1]:3*ll.size()[1], :, :]
        return self.waveletst.inverse(ll, lh, hl, hh, osig)


class FLDQWN(nn.Module):
    def __init__(self, num_classes,
                 first_in_channel, 
                 first_out_channel, 
                 num_level, 
                 kernel_size, 
                 regu_details, 
                 regu_approx,
                 bottleneck,
                 moreconv,
                 wavelet,
                 mode
                 ):
        super(FLDQWN, self).__init__()
        self.num_level = num_level
        self.kernel_size = kernel_size
        self.in_channel = first_in_channel
        self.first_out_channel = first_out_channel
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        self.bottleneck = bottleneck
        self.moreconv = moreconv

        # 第一层：
        self.conv1 = nn.Sequential(
            nn.Conv2d(first_in_channel, first_out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(first_out_channel),
            nn.Tanh(),
            nn.Conv2d(first_out_channel, first_out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(first_out_channel),
            nn.Tanh(),
        )
        
        
        self.wtn = nn.ModuleList()  # 主干网络
        in_channel = first_out_channel
        out_planes = first_out_channel
        if (mode == "CQF_Low_1") or (mode == "CQF_All_1_Filter") or (mode == "CQF_Low_All") or (mode == "CQF_All_All"):
            # 只有第一块训练，其他块不训练,相当于只有一块重复运算
            self.wtn.append(
                LevelWNBlocks(wavelet, in_channel, kernel_size,
                              regu_details, regu_approx, bottleneck, mode=mode)
            )
            
        else:
            for _ in range(num_level):
                self.wtn.append(
                    LevelWNBlocks(wavelet, in_channel, kernel_size,
                                regu_details, regu_approx, bottleneck, mode=mode)
                )
        if self.moreconv:
            out_planes += in_channel*num_level
            self.dsp = self.dsp = downsample_conv_block(level=num_level, in_plane=first_out_channel)  # 下采样网络
        else:
            for i in range(num_level):
                out_planes += in_channel*3
        # self.fc = nn.Linear(out_planes, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(out_planes, 128),       
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)#,
            # nn.Softmax(dim=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        rs = []
        det = []
        x = self.conv1(x)
        if len(self.wtn) == 1:
            for i in range(self.num_level):
                for wtn in self.wtn:
                    x, r, details = wtn(x)
                if self.moreconv:
                    details = self.dsp.blks[i](details)
                rs += [r]
                det += [self.avgpool(details)]
        else:
            for i, wtn in enumerate(self.wtn):
                x, r, details = wtn(x)
                rs += [r]
                if self.moreconv:
                    details = self.dsp.blks[i](details)
                det += [self.avgpool(details)]

        
        aprox = self.avgpool(x)
        det += [aprox]
        x = torch.cat(det,1) 
        x = x.view(-1, x.size()[1])
        return self.fc(x), rs
            
# def pad(signal, padsize, horizontal=True):
#     signal_padded = signal.clone()

#     if horizontal:
#         dim = signal.dim()-1
#         i = padsize // signal.shape[-1]  # recycle
#         ipad = signal.shape[-1]  # recycle padsize
#         jpad = padsize % signal.shape[-1]  # final_padsize

#         # ...
#         for _ in range(i):
#             signal_padded = torch.cat([torch.flip(signal_padded[:,:,:, :ipad], [dim]),
#                                        signal_padded, 
#                                        torch.flip(signal_padded[:,:,:,-ipad:], [dim])],  dim)
#         if jpad==0 and signal.shape[-1] % 2 == 0:
#             return signal_padded
#         if signal.shape[-1] % 2 == 1:

#             # ...
#             signal_padded = torch.cat([torch.flip(signal_padded[:,:,:, :jpad], [dim]),
#                                     signal_padded,
#                                     torch.flip(signal_padded[:,:,:,-jpad-1:], [dim])],  dim)
#         else:
#             # ...
#             signal_padded = torch.cat([torch.flip(signal_padded[:,:,:, :jpad], [dim]),
#                                     signal_padded,
#                                     torch.flip(signal_padded[:,:,:,-jpad:], [dim])],  dim)
#         return signal_padded
#     else:
#         dim = signal.dim()-2
#         i = padsize // signal.shape[-2]  # recycle
#         ipad = signal.shape[-2]
#         jpad = padsize % signal.shape[-2]

#         for _ in range(i):
#             signal_padded = torch.cat([torch.flip(signal_padded[:,:,:ipad, :], [dim]),
#                                        signal_padded, 
#                                        torch.flip(signal_padded[:,:,-ipad:, :], [dim])],  dim)
#         if jpad==0 and signal.shape[-2] % 2 == 0:
#             return signal_padded
#         if signal.shape[-2] % 2 == 1:

#             signal_padded = torch.cat([torch.flip(signal_padded[:,:,:jpad, :], [dim]),
#                                     signal_padded,
#                                     torch.flip(signal_padded[:,:,-jpad-1:, :], [dim])],  dim)
#         else:

#             signal_padded = torch.cat([torch.flip(signal_padded[:,:,:jpad, :], [dim]),
#                                     signal_padded,
#                                     torch.flip(signal_padded[:,:,-jpad:, :], [dim])],  dim)
#         return signal_padded
           

def pad(signal, padsize, horizontal=True):
    signal_padded = signal.clone()

    if horizontal:
        dim = signal.dim()-1
        i = padsize // signal.shape[-1]  # recycle
        ipad = signal.shape[-1]  # recycle padsize
        jpad = padsize % signal.shape[-1]  # final_padsize
        ipadleft = ipad // 2
        ipadright = ipad - ipadleft
        jpadleft = jpad // 2
        jpadright = jpad - jpadleft
        for _ in range(i):

            signal_padded = torch.cat([torch.flip(signal_padded[:,:,:, :ipadleft], [dim]),
                                    signal_padded, 
                                    torch.flip(signal_padded[:,:,:,-ipadright:], [dim])],  dim)
        if jpad==0 and signal.shape[-2] % 2 == 0:
            return signal_padded
        if signal.shape[-1] % 2 == 1:
            signal_padded = torch.cat([torch.flip(signal_padded[:,:,:, :jpadleft], [dim]),
                                    signal_padded,
                                    torch.flip(signal_padded[:,:,:,-jpadright-1:], [dim])],  dim)
        else:
            signal_padded = torch.cat([torch.flip(signal_padded[:,:,:, :jpadleft], [dim]),
                                    signal_padded,
                                    torch.flip(signal_padded[:,:,:,-jpadright:], [dim])],  dim)
        return signal_padded
    else:
        dim = signal.dim()-2
        i = padsize // signal.shape[-2]  # recycle
        ipad = signal.shape[-2]
        jpad = padsize % signal.shape[-2]
        ipadleft = ipad // 2
        ipadright = ipad - ipadleft
        jpadleft = jpad // 2
        jpadright = jpad - jpadleft
        for _ in range(i):
            signal_padded = torch.cat([torch.flip(signal_padded[:,:,:ipadleft, :], [dim]),
                                       signal_padded, 
                                       torch.flip(signal_padded[:,:,-ipadright:, :], [dim])],  dim)
        if jpad==0 and signal.shape[-2] % 2 == 0:
            return signal_padded
        if signal.shape[-2] % 2 == 1:
            signal_padded = torch.cat([torch.flip(signal_padded[:,:,:jpadleft, :], [dim]),
                                    signal_padded,
                                    torch.flip(signal_padded[:,:,-jpadright-1:, :], [dim])],  dim)
        else:
            signal_padded = torch.cat([torch.flip(signal_padded[:,:,:jpadleft, :], [dim]),
                                    signal_padded,
                                    torch.flip(signal_padded[:,:,-jpadright:, :], [dim])],  dim)
        return signal_padded

