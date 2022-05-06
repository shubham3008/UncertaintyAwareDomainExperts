#make model compatible with cifar
#trainable nontrainable layers
#cifar training
#new model weights (finetuned)

import os
from random import seed
import sys
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.autograd import Variable
from mcdropout import ConvNetwork as Network


parser = argparse.ArgumentParser("726_mcdropout")
parser.add_argument('--data', type=str, default='vehicle_classification', help='location of the data corpus')
parser.add_argument('--valid_size', type=float, default=0.25, help='valid size')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--dropout_type', type=str, default='gaussian', help='dropout type')
parser.add_argument('--p', type=float, default=0, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

def main():
  if not torch.cuda.is_available():
    sys.exit(1)

  if args.data == 'animal_classification' or args.data=='cifar':
    CLASSES = 10
  elif args.data=='linnaeus':
    CLASSES = 5
  else:
    CLASSES = 6

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  model = Network((32,32,3),CLASSES,10,'relu',args)
  model = model.cuda()
  save_path = args.data + '_best_weights.pt'
  utils.load(model,save_path)

  for p in model.conv1.parameters():
      p.requires_grad = False

  for p in model.conv2.parameters():
      p.requires_grad = False

  model._modules['linear2'] = torch.nn.Linear(100,10,bias=False)
  model = model.cuda()

  optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
  )
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  train_data = dset.CIFAR10(root='Datasets/cifar', train=True,
                                        download=False, transform=transform)
  valid_data = dset.CIFAR10(root='Datasets/cifar', train=False,
                                        download=False, transform=transform)
  train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
  valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  best_accuracy = 0

  for epoch in range(args.epochs):
    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    scheduler.step()
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    if valid_acc > best_accuracy:
      best_accuracy = valid_acc
      save_path = args.data + '_best_weights_finetuned.pt'
      utils.save(model, os.path.join(save_path))
    utils.save(model, os.path.join('weights.pt'))


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda( non_blocking = True)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()

    prec1 = utils.accuracy(logits, target, topk=(1,))[0]
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)

    if step % args.report_freq == 0:
      print('train %03d %e %f' % (step, objs.avg, top1.avg))

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input = Variable(input).cuda()
      target = Variable(target).cuda(non_blocking = True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1 = utils.accuracy(logits, target, topk=(1,))[0]
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)

    if step % args.report_freq == 0:
      print('valid %03d %e %f' % (step, objs.avg, top1.avg))

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

