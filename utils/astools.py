import time
import torch
import os


class CSVStats(object):
    def __init__(self, resume=False, name="default"):
        self.prec1_train = []
        self.prec1_val = []
        self.prec5_train = []
        self.prec5_val = []
        self.loss_train = []
        self.loss_val = []

        self.resume = resume

        self.out = "runs/%s/stats.csv" % (name)
        if os.path.exists(self.out) and self.resume:
            with open(self.out, "r") as f:
                lines = f.readlines()
                for line in lines[1:]:
                    line = line.strip().split(",")
                    self.prec1_train.append(float(line[0]))
                    self.prec1_val.append(float(line[1]))
                    self.prec5_train.append(float(line[2]))
                    self.prec5_val.append(float(line[3]))
                    self.loss_train.append(float(line[4]))
                    self.loss_val.append(float(line[5]))

    def add(self, p1_train, p1_val, p5_train, p5_val, l_train, l_val):
        self.prec1_train.append(p1_train)
        self.prec1_val.append(p1_val)
        self.prec5_train.append(p5_train)
        self.prec5_val.append(p5_val)
        self.loss_train.append(l_train)
        self.loss_val.append(l_val)

    def write(self):
        with open(self.out, "w") as f:
            f.write('prec1_train,prec1_val,prec5_train,prec5_val,loss_train,loss_val\n')
            for i in range(len(self.prec1_val)):
                f.write("{:.5f},{:.5f},{:.5f},{:.5f},{},{}\n".format(
                    self.prec1_train[i], self.prec1_val[i],
                    self.prec5_train[i], self.prec5_val[i],
                    self.loss_train[i], self.loss_val[i]))



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, lossfunc, optimizer, epoch, print_freq, tpk, model_name):
    batch_time = AverageMeter()
    losses_total = AverageMeter()
    losses_class = AverageMeter()
    losses_regu1 = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()

    model.train()  # train mode

    end = time.time()  # 计时

    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        if model_name == 'fldqwn':
            output, regus1 = model(input_var)  # 输出结果，正则损失
            loss_class = lossfunc(output, target_var)  # 损失
            loss_regu1 = sum(regus1)
            loss_total = loss_class + loss_regu1
            losses_class.update(loss_class.item(), input.size(0))  # 预测损失更新
            losses_regu1.update(loss_regu1.item(), input.size(0))  # 正则损失更新
            losses_total.update(loss_total.item(), input.size(0))  # 总损失更新
        else:
            output = model(input_var)
            loss_class = lossfunc(output, target_var)
            loss_total = loss_class
            losses_class.update(loss_class.item(), input.size(0))
            losses_regu1.update(0.0, input.size(0))
            losses_regu1.update(0.0, input.size(0))
            losses_total.update(loss_total.item(), input.size(0))


        prec1 = accuracy(output.data, target, topk=(1,))[0]  # 预测值==target
        preck = accuracy(output.data, target, topk=(tpk,))[0]  # 前5个预测值有正确的target

        top1.update(prec1.item(), input.size(0))  # prec1更新
        topk.update(preck.item(), input.size(0))  # prec5更新

        # 反向传播
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # 计算用时
        batch_time.update(time.time() - end)
        end = time.time()

        # 输出训练信息
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]  '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Loss (Class) {loss_class.val:.4f} ({loss_class.avg:.4f})  '
                  'Loss (Regu1) {loss_regu1.val:.4f} ({loss_regu1.avg:.4f})  '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
                  'Prec@k {topk.val:.3f} ({topk.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss_class=losses_class, loss_regu1=losses_regu1, 
                      top1=top1,topk=topk))

    return (top1.avg, topk.avg, losses_total.avg)


def validate(val_loader, model, criterion, epoch, print_freq, tpk, model_name):
    batch_time = AverageMeter()
    losses_total = AverageMeter()
    losses_class = AverageMeter()
    losses_regu1 = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()

    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            if model_name == 'fldqwn':
                output, regus1 = model(input_var)
                loss_class = criterion(output, target_var)
                loss_regu1 = sum(regus1)
                loss_total = loss_class + loss_regu1
                losses_class.update(loss_class.item(), input.size(0))
                losses_total.update(loss_total.item(), input.size(0))
                losses_regu1.update(loss_regu1.item(), input.size(0))
            else:
                output = model(input_var)
                loss_class = criterion(output, target_var)
                loss_total = loss_class
                losses_class.update(loss_class.item(), input.size(0))
                losses_total.update(loss_total.item(), input.size(0))
                losses_regu1.update(0.0, input.size(0))


            prec1 = accuracy(output.data, target, topk=(1,))[0]
            preck = accuracy(output.data, target, topk=(tpk,))[0]

            top1.update(prec1.item(), input.size(0))
            topk.update(preck.item(), input.size(0))


            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]  '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'Loss (Class) {loss_class.val:.4f} ({loss_class.avg:.4f})  '
                      'Loss (Regu1) {loss_regu1.val:.4f} ({loss_regu1.avg:.4f})  '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
                      'Prec@k {topk.val:.3f} ({topk.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss_class=losses_class, loss_regu1=losses_regu1,
                          top1=top1, topk=topk))

    return (top1.avg, topk.avg, losses_total.avg)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, pathname, filename='checkpoint.pth.tar' ):
    directory = "runs/%s/" % (pathname)

    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)

    if is_best:
        best_name=directory + 'model_best.pth.tar'
        torch.save(state, best_name)


