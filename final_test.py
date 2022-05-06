#comparison
#individual models validation accuracy
#algorithm (for any image -> feature extractor -> vector -> distances -> weights -> all 3 probabilities multiplied by weights)
# d1, d2, d3 -> exp(d1)/(sum(exp)) = w1
#summation pi*wi -> p_final

#make model compatible with cifar
#trainable nontrainable layers
#cifar training
#new model weights (finetuned)

import os
from random import seed
import sys
import numpy as np
import torch
import math
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
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
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

  feature_extractor = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet32", pretrained=True)
  feature_extractor = feature_extractor.cuda()

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  data1_prototype = [-0.98613995,0.09747741,0.27661768,0.47434798,0.31566167,-0.577094,-0.2550221,0.67170495,-0.03842954,0.47023973,1.0537182,1.0033482,-0.9540847,-0.3564511,0.13654381,0.41366646,-0.00537106,-0.5221664,0.46272668,0.7639119,-0.62662476,-0.41356668,0.547785,-0.2774156,-0.5964714,0.68046427,0.0459709,-0.5073568,-0.15711589,1.6278092,-1.253913,0.4587141,1.2303742,0.8920671,-0.33306682,-0.1058125,-0.29673848,0.33790454,0.16170554,-0.06074653,1.115609,0.41989517,0.7206063,-1.0491363,0.7679247,0.63315344,0.73177975,-0.9522387,-0.57035124,-0.7867017,0.6552762,0.7087146,-1.0799383,-1.6201018,-0.70793563,0.21788473,-0.53739995,0.73253554,-1.1187648,-0.34290692,-0.18628542,-0.8664017,-0.3203415,-0.01652408,0.77858394,1.1610632,0.7461819,1.0223414,-0.40621582,0.1261905,-0.84492254,-1.5022625,0.94301933,0.878149,0.53772044,0.33394372,-0.6459165,0.43032733,0.5415599,0.5994603,-0.28750262,-1.4995544,0.37468952,-0.9214844,0.47917327,0.13950704,-0.80172247,0.7006362,-0.20501085,0.19999804,0.3868829,0.39765182,-0.4459753,0.2410649,-2.0693653,-1.9285336,0.38540223,-0.73352987,0.462433,0.07809339]
  data2_prototype = [-1.0263987,0.07313988,0.2254778,0.49099633,0.3846036,-0.61608505,-0.23017672,0.6639918,-0.07021604,0.38798106,1.0111016,1.0311806,-0.93374294,-0.33517894,0.10401502,0.38587907,-0.05032535,-0.52604944,0.470906,0.7985462,-0.59373087,-0.4868434,0.52248704,-0.26954117,-0.596052,0.73250496,0.04070656,-0.45690885,-0.158192,1.5598079,-1.2044939,0.463123,1.2068723,0.8826794,-0.2598371,-0.12610884,-0.25105673,0.325682,0.16143613,-0.01012049,1.053657,0.40633297,0.78096646,-1.0101721,0.7172375,0.58590126,0.709196,-0.9102809,-0.54634696,-0.7552176,0.6713848,0.75347435,-1.0507743,-1.6043714,-0.8109833,0.18240206,-0.6107168,0.7191892,-1.1036901,-0.4139061,-0.12257266,-0.88719064,-0.26941308,0.08973584,0.7836659,1.1472456,0.7184438,1.0356234,-0.37763068,0.05725217,-0.86741316,-1.4420433,0.97817945,0.8677586,0.5642394,0.33498415,-0.6688571,0.34305382,0.5428824,0.5780768,-0.3295777,-1.51728,0.42915374,-0.9606053,0.51691586,0.09535947,-0.7905973,0.793199,-0.21287468,0.20712025,0.40282485,0.34600487,-0.40989837,0.24026978,-1.9824402,-1.9005921,0.3853557,-0.76941776,0.4495183,0.11580727]
  data3_prototype = [-0.9946039,0.07018023,0.2530198,0.46623605,0.30839744,-0.58158904,-0.25200805,0.65614015,-0.04067389,0.40763834,1.04634,1.0426768,-0.9449535,-0.36337826,0.11238802,0.38413167,-0.02224402,-0.51417685,0.43254486,0.7758846,-0.6081095,-0.45865577,0.53436065,-0.27008513,-0.61915475,0.67602885,0.03031754,-0.5127892,-0.13830368,1.562457,-1.2139182,0.4566201,1.2356312,0.8721527,-0.29912803,-0.11192147,-0.21900111,0.3433709,0.12463661,-0.02249072,1.104904,0.3887503,0.7296397,-1.0300573,0.7322076,0.5859133,0.76131576,-0.92691654,-0.567688,-0.7354878,0.6714872,0.7575502,-1.0837634,-1.5816568,-0.70961523,0.17857872,-0.5809128,0.7056216,-1.1080862,-0.39714223,-0.13293135,-0.8271112,-0.2572801,0.01752251,0.80521446,1.1395794,0.71058524,1.0262585,-0.371209,0.07252339,-0.8338934,-1.4510121,0.8953035,0.8885495,0.53404003,0.31771794,-0.62726355,0.36471716,0.5431619,0.5934879,-0.29436755,-1.5569042,0.4363224,-0.925355,0.5459586,0.1006933,-0.797277,0.7726057,-0.21178173,0.18881117,0.37584746,0.36044255,-0.39615592,0.22509874,-1.9975555,-1.916232,0.36372682,-0.7453488,0.47961628,0.08488749]

  data1_prototype = torch.tensor(data1_prototype).float().cuda()
  data2_prototype = torch.tensor(data2_prototype).float().cuda()
  data3_prototype = torch.tensor(data3_prototype).float().cuda()

  model1 = Network((32,32,3),10,10,'relu',args)
  model1 = model1.cuda()
  save_path = 'vehicle_classification_best_weights_finetuned.pt'
  utils.load(model1,save_path)

  model2 = Network((32,32,3),10,10,'relu',args)
  model2 = model2.cuda()
  save_path = 'animal_classification_best_weights_finetuned.pt'
  utils.load(model2,save_path)

  model3 = Network((32,32,3),10,10,'relu',args)
  model3 = model3.cuda()
  save_path = 'linnaeus_best_weights_finetuned.pt'
  utils.load(model3,save_path)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  valid_data = dset.CIFAR10(root='Datasets/cifar', train=False,
                                        download=False, transform=transform)
  valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

  best_accuracy = 0

  for i,model in enumerate([model1,model2,model3]):
        valid_acc, valid_obj = infer(valid_queue, model, criterion,i)

  verdict_acc, verdict_obj = verdict_infer(valid_queue, model1, model2, model3, feature_extractor, criterion, data1_prototype, data2_prototype, data3_prototype)

def infer(valid_queue, model, criterion,i):
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


  print('valid [model %03d] %e %f' % (i+1, objs.avg, top1.avg))
  return top1.avg, objs.avg

def verdict_infer(valid_queue, model1, model2, model3, feature_extractor, criterion, data_prototype1, data_prototype2, data_prototype3):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input = Variable(input).cuda()
      target = Variable(target).cuda(non_blocking = True)

    logits1 = model1(input)
    logits2 = model2(input)
    logits3 = model3(input)
    feature = feature_extractor(input)
    d1 = torch.exp(-torch.norm(feature-data_prototype1,dim=1)).view(-1,1)
    d2 = torch.exp(-torch.norm(feature-data_prototype2,dim=1)).view(-1,1)
    d3 = torch.exp(-torch.norm(feature-data_prototype3,dim=1)).view(-1,1)
    logits = (d1 * logits1 + d2 * logits2 + d3 * logits3)/(d1+d2+d3)
    # print(d1,d2,d3)
    # print(logits1,logits2,logits3,logits)
    loss = criterion(logits, target)
    prec1 = utils.accuracy(logits, target, topk=(1,))[0]
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)

  print('valid [verdict] %e %f' % (objs.avg, top1.avg))
  return top1.avg, objs.avg

if __name__ == '__main__':
  main() 

