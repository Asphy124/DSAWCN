import time
import torch
import os
import sys

class CSVStats(object):
    def __init__(self, resume=False, name="default"):
        self.prec1_train = []
        self.prec1_val = []
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
                    self.loss_train.append(float(line[2]))
                    self.loss_val.append(float(line[3]))

    def add(self, p1_train, p1_val, l_train, l_val):
        self.prec1_train.append(p1_train)
        self.prec1_val.append(p1_val)
        self.loss_train.append(l_train)
        self.loss_val.append(l_val)

    def write(self):
        with open(self.out, "w") as f:
            f.write('prec1_train,prec1_val,loss_train,loss_val\n')
            for i in range(len(self.prec1_val)):
                f.write("{:.5f},{:.5f},{},{}\n".format(
                    self.prec1_train[i], self.prec1_val[i],
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


def train(train_loader, model, lossfunc, optimizer, model_name):
    losses_total = AverageMeter()
    losses_class = AverageMeter()
    losses_regu1 = AverageMeter()
    top1 = AverageMeter()

    model.train()  # train mode

    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        if model_name == 'fldqwn':
            output, regus1 = model(input_var)
            loss_class = lossfunc(output, target_var)
            loss_regu1 = sum(regus1)
            loss_total = loss_class + loss_regu1
            losses_class.update(loss_class.item(), input.size(0))
            losses_regu1.update(loss_regu1.item(), input.size(0))
            losses_total.update(loss_total.item(), input.size(0))
        else:
            output = model(input_var)
            loss_class = lossfunc(output, target_var)
            loss_total = loss_class
            losses_class.update(loss_class.item(), input.size(0))
            losses_regu1.update(0.0, input.size(0))
            losses_regu1.update(0.0, input.size(0))
            losses_total.update(loss_total.item(), input.size(0))

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        progress_bar(i, len(train_loader), 'Loss1: %.3f | Loss2: %.3f | Acc: %.3f%%'
                     % (losses_class.avg, losses_regu1.avg, top1.avg))

    return (top1.avg, losses_total.avg)


def validate(val_loader, model, criterion, model_name):
    losses_total = AverageMeter()
    losses_class = AverageMeter()
    losses_regu1 = AverageMeter()
    top1 = AverageMeter()

    model.eval()

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
            top1.update(prec1.item(), input.size(0))

            progress_bar(i, len(val_loader), 'Loss1: %.3f | Loss2: %.3f | Acc: %.3f%%'
                     % (losses_class.avg, losses_regu1.avg, top1.avg))

    return (top1.avg, losses_total.avg)


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


term_width = 80
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

