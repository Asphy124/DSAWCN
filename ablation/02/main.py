import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time  
import torchvision
import os
import model.fldqwn as fldqwn
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
parser.add_argument("--lrdecay", nargs='+', type=int, default=[30,60,90,120]) 
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
parser_despawn.add_argument('--mode', choices=['Stable', 'Free'], default='Free')


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



train_loader, val_loader, num_classes = load_data(args.data_name, 
                                                  args.gcn,
                                                  args.split_data,
                                                  args.batch_size)

name = args.data_name + '_' + args.name


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

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=0.9, nesterov=True,
                            weight_decay=1e-4)


print("Number of model parameters            : {:,}".format(
    sum([p.data.nelement() for p in model.parameters()])))
print("Number of *trainable* model parameters: {:,}".format(
    sum(p.numel() for p in model.parameters() if p.requires_grad)))

best_prec1 = 0

for epoch in range(args.epochs):
    t0 = time.time()
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

