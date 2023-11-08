import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time  
import torchvision
import os
import model.fldqwn as fldqwn
import model.dawn as dawn
import model.wcnn as wcnn
import math
from utils.astools import CSVStats, train, validate, save_checkpoint
from utils.load_data import load_data
parser = argparse.ArgumentParser(description='Training cammands')

parser.add_argument('--data_name', type=str, default='bark-20', help='dataset name')
parser.add_argument('--gcn', type=bool, default=False, help='gcn')
parser.add_argument('--split_data', type=float, default=0.20, help='split data')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
parser.add_argument('--epochs', type=int, default=300, help='learning epochs')
parser.add_argument('--name', default='fldqwn', type=str)
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument("--lrdecay", nargs='+', type=int, default=[30,60,90,120])  #除39外使用此衰减值
# parser.add_argument("--lrdecay", nargs='+', type=int, default=[60, 120, 160, 200])  #除39外使用此衰减值
parser.add_argument('--drop', default=3, type=int, help='drop learning rate')


subparsers = parser.add_subparsers(dest="model")
parser_despawn = subparsers.add_parser('fldqwn')
parser_despawn.add_argument('--first_out_channel', default=64, type=int, help='first out channel')
parser_despawn.add_argument('--num_level', default=5, type=int, help='number of level')
parser_despawn.add_argument('--kernel_size', default=8, type=int, help='wavelet kernel_size')
parser_despawn.add_argument('--regu_details', default=0.1, type=float, help='regu_details')
parser_despawn.add_argument('--regu_approx', default=0.1, type=float, help='regu_approx')
parser_despawn.add_argument('--bottleneck', default=True, type=bool, help='bottleneck')
parser_despawn.add_argument('--moreconv', default=True, type=bool, help='moreconv')
parser_despawn.add_argument('--wavelet', default='db4', type=str, help='wavelet')
parser_despawn.add_argument('--mode', choices=['Stable', 'CQF_Low_1', 'CQF_All_1_Filter', 'CQF_Low_All', 'CQF_All_All',
                                               'Layer_Low_1', 'Layer_All_1_Filter', 'Layer_Low_All', 'Free'], default='Free')


parser_resnet18 = subparsers.add_parser('resnet18')
parser_vgg16 = subparsers.add_parser('vgg16')
parser_densenet = subparsers.add_parser('densenet')
parser_squeezenet = subparsers.add_parser('squeezenet')
parser_mobilenet = subparsers.add_parser('mobilenet')
parser_shufflenet = subparsers.add_parser('shufflenet')
parser_efficientnet = subparsers.add_parser('efficientnet')

parser_dawn = subparsers.add_parser('dawn')
parser_dawn.add_argument("--regu_details", default=0.1, type=float)
parser_dawn.add_argument("--regu_approx", default=0.1, type=float)
parser_dawn.add_argument("--levels", default=4, type=int)
parser_dawn.add_argument("--first_conv", default=32, type=int)
parser_dawn.add_argument("--classifier", default='mode1', choices=['mode1', 'mode2','mode3'])
parser_dawn.add_argument("--kernel_size", type=int, default=3)
parser_dawn.add_argument("--no_bootleneck", default=False, action='store_true')
parser_dawn.add_argument("--share_weights", default=False, action='store_true')
parser_dawn.add_argument("--simple_lifting", default=False, action='store_true')
parser_dawn.add_argument("--haar_wavelet", default=False, action='store_true')
parser_dawn.add_argument('--warmup', default=False, action='store_true')

parser_scatter = subparsers.add_parser('scatter')
parser_scatter.add_argument('--scat', default=2, type=int,help='scattering scale, j=0 means no scattering')
parser_scatter.add_argument('--N', default=32, type=int,help='size of the crop')
parser_scatter.add_argument('--classifier', type=str, default='WRN',help='classifier model [WRN, mlp, linear]')
parser_scatter.add_argument('--mode', type=int, default=1,help='scattering 1st or 2nd order')
parser_scatter.add_argument('--blocks', type=int, default=2,help='for WRN number of blocks of layers: n ')
parser_scatter.add_argument('--use_avg_pool', default=False, action='store_true', help='use avg pooling before the classifier')

parser_wcnn = subparsers.add_parser('wcnn')
parser_wcnn.add_argument("--wavelet", choices=['haar', 'db2', 'lifting'])
parser_wcnn.add_argument("--levels", default=4, type=int)

args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch, inv_drop):
    lr = args.lr
    drop = 1.0 / inv_drop
    factor = math.pow(
        drop, sum([1.0 if epoch >= e else 0.0 for e in args.lrdecay]))
    lr = lr * factor
    print("Learning rate: ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 载入数据集
train_loader, val_loader, num_classes = load_data(args.data_name, 
                                                  args.gcn,
                                                  args.split_data,
                                                  args.batch_size)

name = args.data_name + '_' + args.name
# 载入模型
"""
DeSpawn2D可训练参数量 = kernel_size * in_channel * f(mode, num_level)
mode:
    "Stable": 固定  可训练参数量6 * 3 * 0 = 0

    "CQF_Low_1": 一块的一子块  的低通滤波器训练，高通滤波器通过公式求得，子块和块共享  [可训练参数量]kernel_size * in_channel * 1
    "CQF_All_1_Filter": 一块的第一子块  的低高通滤波器训练，子块和块共享  [可训练参数量]kernel_size * in_channel * 2
    "CQF_Low_All": 一块的所有子块  的低通滤波器训练，高通滤波器通过公式求得，块共享  [可训练参数量]kernel_size * in_channel * 3
    "CQF_All_All": 一块的所有子块的高低通滤波器训练，块共享  [可训练参数量]kernel_size * in_channel * 6

    "Layer_Low_1": 每块的一子块  的低通滤波器训练，高通滤波器通过公式求得，子块共享  [可训练参数量]kernel_size * in_channel * num_level * 1
    "Layer_All_1_Filter": 每块的第一子块  的低高通滤波器训练，子块共享  [可训练参数量]kernel_size * in_channel * num_level * 2
    "Layer_Low_All": 每块的所有子块的  低通滤波器训练，高通滤波器通过公式求得  [可训练参数量]kernel_size * in_channel * num_level * 3
    "Free": 所有滤波器自由训练  [可训练参数量]kernel_size * in_channel * num_level * 6
"""
if args.model == 'fldqwn':
    model = fldqwn.FLDQWN(num_classes=num_classes, 
                            first_in_channel=3,
                            first_out_channel=args.first_out_channel,
                            num_level=args.num_level,
                            kernel_size=args.kernel_size,
                            regu_details=args.regu_details,
                            regu_approx=args.regu_approx,
                            bottleneck=args.bottleneck,
                            moreconv=args.moreconv,
                            wavelet=args.wavelet,
                            mode=args.mode)
elif args.model == 'resnet18':
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, num_classes)

elif args.model == 'vgg16':
    model = torchvision.models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, num_classes)

elif args.model == 'densenet':
    model = torchvision.models.densenet121(pretrained=False)
    model.classifier = nn.Linear(1024, num_classes)

elif args.model == 'squeezenet':
    model = torchvision.models.squeezenet1_1(pretrained=False)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

elif args.model == 'mobilenet':
    model = torchvision.models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(1280, num_classes)

elif args.model == 'shufflenet':
    model = torchvision.models.shufflenet_v2_x1_0(pretrained=False)
    model.fc = nn.Linear(1024, num_classes)

elif args.model == 'efficientnet':
    model = torchvision.models.efficientnet_b0(pretrained=False)
    model._fc = nn.Linear(1280, num_classes)

elif args.model == 'dawn':
    model = dawn.DAWN(num_classes, big_input=True,
                    first_conv=args.first_conv,
                    number_levels=args.levels,
                    kernel_size=args.kernel_size,
                    no_bootleneck=args.no_bootleneck,
                    classifier=args.classifier,
                    share_weights=args.share_weights,
                    simple_lifting=args.simple_lifting,
                    COLOR=True,
                    regu_details=args.regu_details,
                    regu_approx=args.regu_approx,
                    haar_wavelet=args.haar_wavelet
                    )
elif args.model == 'wcnn':
    model = wcnn.WCNN(
        num_classes, big_input=True, wavelet=args.wavelet, levels=args.levels)

elif args.model == 'scatter':
    from kymatio import Scattering2D
    from model.scatter.Scatter_WRN import ScatResNet
    
    scattering = Scattering2D(J=args.scat, shape=(args.N, args.N), max_order=args.mode)
    scattering = scattering.cuda()
    model = ScatResNet(args.scat, 256, num_classes, args.classifier, args.mode)

if args.resume:
    directory = "runs/%s/" % (name)
    filename = directory + 'checkpoint.pth.tar'
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

model = model.cuda()

cudnn.benchmark = True
csv_logger = CSVStats(resume=args.resume, name=name)
lossfunc = nn.CrossEntropyLoss().cuda()
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
#                             betas=(0.9,0.999), eps=1e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=0.9, nesterov=True,
                            weight_decay=1e-4)


print("Number of model parameters            : {:,}".format(
    sum([p.data.nelement() for p in model.parameters()])))
print("Number of *trainable* model parameters: {:,}".format(
    sum(p.numel() for p in model.parameters() if p.requires_grad)))

best_prec1 = 0
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
#                                            factor=0.1, patience=15, 
#                                            verbose=True, threshold=0.0001, 
#                                            threshold_mode='rel', 
#                                            cooldown=0, min_lr=0, eps=1e-08)
for epoch in range(args.epochs):
    t0 = time.time()  # 计时
    adjust_learning_rate(optimizer, epoch, args.drop)
    prec1_train, loss_train = train(
        train_loader, model, lossfunc, optimizer, model_name=args.model)
    torch.cuda.empty_cache()
    prec1_val, loss_val = validate(
            val_loader, model, lossfunc, model_name=args.model)
    csv_logger.add(prec1_train, prec1_val, loss_train, loss_val)

    is_best = prec1_val > best_prec1
    best_prec1 = max(prec1_val, best_prec1)

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
    }, is_best, name)

    csv_logger.write()

    # Final print
    print('Epoch {} : Train[{:.3f} %, {:.3f} loss] Val [{:.3f} %, {:.3f} loss] Best: {:.3f} %'.format(
        epoch, prec1_train, loss_train, prec1_val, loss_val, best_prec1))

print('Best accuracy: ', best_prec1)

# end

