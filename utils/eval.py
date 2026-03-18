import torch
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision.transforms import v2

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count+1e-8)


# evaluate
def accuracy(y_pred, y_actual, topk=(1, ), return_tensor=False):
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        if return_tensor:
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(correct_k.item() * 100.0 / batch_size)
    return res


def evaluate_cls_acc(dataloader, model, dev, topk=(1,)):
    model.eval()
    test_loss = AverageMeter()
    test_loss.reset()
    test_accuracy = AverageMeter()
    test_accuracy.reset()

    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc='evaluating')):
            if type(sample) is dict:
                x = sample['data'].to(dev)
                y = sample['label'].to(dev)
            else:
                # x, y, _ = sample
                x, y,_,_ = sample
                x, y = x.to(dev), y.to(dev)
            output = model(x)
            logits = output['logits'] if type(output) is dict else output
            loss = torch.nn.functional.cross_entropy(logits, y)
            test_loss.update(loss.item(), x.size(0))
            acc = accuracy(logits, y, topk)
            test_accuracy.update(acc[0], x.size(0))
    return {'accuracy': test_accuracy.avg, 'loss': test_loss.avg}

def evaluate_cls_acc_TTA(dataloader, model, dev, topk=(1,)):
    model.eval()
    test_loss = AverageMeter()
    test_loss.reset()
    test_accuracy = AverageMeter()
    test_accuracy.reset()
    # test_transform = torchvision.transforms.Compose([
    #         torchvision.transforms.ToPILImage(),
    #         torchvision.transforms.RandomHorizontalFlip(),
    #         torchvision.transforms.RandomCrop(size=448),
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #     ])

    test_transform = v2.Compose([
        v2.ToPILImage(),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(size=448),
        v2.ToTensor(),
        v2.Normalize(mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)),
    ])

    # 对图像应用多次增强和预测
    # for _ in range(3):
    #     augmented_image = transform(image)
    #     augmented_image = augmented_image.unsqueeze(0)  # 增加一个维度作为批次
    #     with torch.no_grad():
    #         # 切换模型为评估模式，确保不执行梯度计算
    #         model.eval()
    #         # 使用增强的图像进行预测
    #         output = model(augmented_image)
    #         _, predicted = torch.max(output.data, 1)
    #         predictions.append(predicted.item())

    # 执行多数投票并返回最终预测结果
    # final_prediction = np.bincount(predictions).argmax()
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc='evaluating')):
            if type(sample) is dict:
                x = sample['data'].to(dev)
                y = sample['label'].to(dev)
            else:
                # x, y, _ = sample
                x, y,_,_ = sample
                x, y = x.to(dev), y.to(dev)
            final_logits = 0
            predictions = []
            for _ in range(3):
                augmented_image = torch.stack([test_transform(img) for img in x])
                # augmented_image = augmented_image.unsqueeze(0)  # 增加一个维度作为批次
                with torch.no_grad():
                    # 切换模型为评估模式，确保不执行梯度计算
                    model.eval()
                    # 使用增强的图像进行预测
                    output = model(augmented_image.cuda())
                    logits = output['logits'] if type(output) is dict else output
                    # _, predicted = torch.max(output.data, 1)
                    final_logits+=logits
            final_logits= final_logits/3
            loss = torch.nn.functional.cross_entropy(final_logits, y)
            test_loss.update(loss.item(), x.size(0))
            acc = accuracy(final_logits, y, topk)
            test_accuracy.update(acc[0], x.size(0))
    return {'accuracy': test_accuracy.avg, 'loss': test_loss.avg}



def evaluate_relabel_pr(given_labels, corrected_labels):
    precision = 0.0
    recall = 0.0
    # TODO: code for evaluation of relabeling (precision, recall)
    return {'relabel-precision': precision, 'relabel-recall': recall}
