import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models as teacher_models
# import models
import models as student_models
from imbalance_data.lt_data import LT_Dataset
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from losses import *

model_names = sorted(name for name in teacher_models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(teacher_models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='iNaturalist18', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet32)')
# resnext50_32x4d resnet10
parser.add_argument('--loss_type', default="BKD", type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='D:\\PycharmProjects\\xjy\\BalancedKnowledgeDistillation-main\\student_checkpoint_iNaturalist\\iNaturalist18_resnet50_BKD_None_exp_0.01_T2.0_0_0620one\\ckpt.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log', type=str, default='student_log_iNaturalist')
parser.add_argument('--root_model', type=str, default='student_checkpoint_iNaturalist')
parser.add_argument('--T', '--temperature', default=2.0, type=float,
                    metavar='N',
                    help='distillation temperature')
parser.add_argument('--data_aug', default="CMO", type=str, help='data augmentation type',
                    choices=('vanilla', 'CMO'))
parser.add_argument('--weighted_alpha', default=1, type=float, help='weighted alpha for sampling probability (q(1,k))')
parser.add_argument('--start_data_aug', default=3, type=int, help='start epoch for aug')
parser.add_argument('--end_data_aug', default=3, type=int, help='how many epochs to turn off aug')
parser.add_argument('--mixup_prob', default=0.5, type=float, help='mixup probability')
parser.add_argument('--beta', default=1, type=float, help='hyperparam for beta distribution')

parser.add_argument('--root', default='D:\\Datasets', help='dataset setting')

parser.add_argument('--alpha', default=1.0, type=float, metavar='M',
                    help='alpha')
parser.add_argument('--model_dir', type=str, default=None)
best_acc1 = 0


def main():
    args = parser.parse_args()
    if not os.path.exists(args.root_log):
        os.mkdir(args.root_log)
    # args.root_log = os.path.join(args.root_log, args.dataset)
    if not os.path.exists(args.root_log):
        os.mkdir(args.root_log)
    args.store_name = '_'.join(
        [args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type,
         str(args.imb_factor), 'T' + str(args.T), args.exp_str, '0620one'])
         # str(args.imb_factor), 'T' + str(args.T), args.exp_str])
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.gpu = 0

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global train_cls_num_list
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 8142
    args.train_rule = 'Reweight'
    teacher_model = teacher_models.__dict__[args.arch](num_classes=num_classes)
    student_model = student_models.__dict__[args.arch](num_classes=num_classes)
    # teacher_model = getattr(models, args.arch)(pretrained=False)

    # student_model = getattr(models, args.arch)(pretrained=False)

    teacher_model = load_network(teacher_model, args)

    args.num_classes = num_classes

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        teacher_model = teacher_model.to(args.gpu)
        student_model = student_model.to(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        teacher_model = torch.nn.DataParallel(teacher_model).cuda()
        student_model = torch.nn.DataParallel(student_model).cuda()

    optimizer = torch.optim.SGD(student_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            student_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = LT_Dataset(args.root, 'iNaturalist18/iNaturalist18_train.txt', transform_train,
                               use_randaug=False)
    val_dataset = LT_Dataset(args.root, 'iNaturalist18/iNaturalist18_val.txt', transform_val)

    num_classes = len(np.unique(train_dataset.targets))
    assert num_classes == 8142

    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    train_cls_num_list = np.array(cls_num_list)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    weighted_train_loader = None
    if args.data_aug == 'CMO':
        cls_weight = 1.0 / (np.array(cls_num_list) ** args.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print(samples_weight)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),
                                                                  replacement=True)
        weighted_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                            num_workers=args.workers, pin_memory=True,
                                                            sampler=weighted_sampler)

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        if args.train_rule == 'None':
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)
            per_cls_weights = None
        elif args.train_rule == 'Reweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')

        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'KD':

            criterion = KDLoss(cls_num_list=cls_num_list, T=args.T, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'BKD':
            criterion = BKDLoss(cls_num_list=cls_num_list, T=args.T, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'Focal':
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        train(train_loader, teacher_model, student_model, criterion, optimizer, epoch, args, log_training, tf_writer, weighted_train_loader)

        # evaluate on validation set
        acc1 = validate(val_loader, teacher_model, student_model, criterion, epoch, args, log_testing, tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': student_model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def train(train_loader, teacher_model, student_model, criterion, optimizer, epoch, args, log, tf_writer, weighted_train_loader=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')


    # switch to train mode
    student_model.train()
    teacher_model.train()

    end = time.time()

    if args.data_aug == 'CMO' and args.start_data_aug < epoch < (args.epochs - args.end_data_aug):
        inverse_iter = iter(weighted_train_loader)

    for i, (input, target) in enumerate(train_loader):
        input_g = input
        ##add
        if args.data_aug == 'CMO' and args.start_data_aug < epoch < (args.epochs - args.end_data_aug):
            try:
                input2, target2 = next(inverse_iter)
            except:
                inverse_iter = iter(weighted_train_loader)
                input2, target2 = next(inverse_iter)
            input2 = input2[:input.size()[0]]
            target2 = target2[:target.size()[0]]
            input2 = input2.cuda(args.gpu, non_blocking=True)
            target2 = target2.cuda(args.gpu, non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        r = np.random.rand(1)
        # loss_aug=None
        # bbx1, bby1, bbx2, bby2 = 0
        # global bbx1, bby1, bbx2, bby2
        if args.data_aug == 'CMO' and args.start_data_aug < epoch < (args.epochs - args.end_data_aug) and r < args.mixup_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)

            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio

            # compute output

            # loss_aug = criterion(output_aug, target) * lam + criterion(output_aug, target2) * (1. - lam)
            # print('111111111')
            # loss_aug = loss_aug.cuda(args.gpu, non_blocking=True)
        # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        # output_aug = student_model(input)
        # compute output
        with torch.no_grad():
            teacher_output = teacher_model(input)
        output = student_model(input)

        alpha = args.alpha
        if 'KD' in args.loss_type:
            loss, kd = criterion(output, teacher_output, target, alpha)
            # loss, kd = criterion(output, teacher_output, output_aug, target, alpha, lam, target2 )
            # loss = loss + loss_aug
        else:
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      # 'KDLoss {kd_loss.val:.4f} ({kd_loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, #kd_loss=kd_loss,
                lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, teacher_model, student_model, criterion, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')


    # switch to evaluate mode
    teacher_model.eval()
    student_model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.no_grad():
                teacher_output = teacher_model(input)
            output = student_model(input)

            alpha = 0

            # if 'hh' in args.loss_type:
            #     loss, kd = criterion(output, teacher_output, target, alpha)
            if 'KD' in args.loss_type:
                loss, kd = criterion(output, teacher_output, target, alpha)
            else:
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
        flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(epoch)
        print(output)
        print(out_cls_acc)
        many_shot = train_cls_num_list > 100
        medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list > 20)
        few_shot = train_cls_num_list <= 20
        print("many avg, med avg, few avg", float(sum(cls_acc[many_shot]) * 100 / sum(many_shot)),
              float(sum(cls_acc[medium_shot]) * 100 / sum(medium_shot)),
              float(sum(cls_acc[few_shot]) * 100 / sum(few_shot)))

        many_avg=float(sum(cls_acc[many_shot]) * 100 / sum(many_shot))
        mid_avg=float(sum(cls_acc[medium_shot]) * 100 / sum(medium_shot))
        few_avg=float(sum(cls_acc[few_shot]) * 100 / sum(few_shot))


        if log is not None:
            utput = ('Epoch: [{0}]\t'.format(epoch))  # TODO
            log.write(utput + '\n')
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')

            many_avgo = ('many_avg: {0}\t'.format(many_avg))
            log.write(many_avgo)

            mid_avgo = ('mid_avg: {0}\t'.format(mid_avg))
            log.write(mid_avgo)

            few_avgo = ('few_avg: {0}\t'.format(few_avg))
            log.write(few_avgo +'\n')

            log.flush()


        tf_writer.add_scalar('loss/test_' + flag, losses.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)

    return top1.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 160:
        lr = args.lr * 0.01
    elif epoch > 75:
        lr = args.lr * 0.1
    else:
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_network(network, args):

    save_path = 'teacher_checkpoint/iNaturalist18_resnet50_CE_None_exp_0.02_0/ckpt.pth.tar'
    if args.model_dir:
        save_path = os.path.join(args.model_dir, 'ckpt.pth.tar')
    print(save_path)
    network = nn.DataParallel(network)
    network.load_state_dict(torch.load(save_path, map_location='cuda:0')['state_dict'])
    return network

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def rand_bbox_withcenter(size, lam, cx, cy):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


if __name__ == '__main__':
    main()