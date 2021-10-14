import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from model import MobileNetV1
CHECKPOINT_PATH = "checkpoint"
SAVE_EPOCH = 10
MILESTONES = [60, 120, 160]

def train(epoch, net):
    # f = open("time.txt","a")
    # f.write("开始时间："+strftime("%Y-%m-%d %H:%M:%S", localtime()))
    # f.write("\n")
    # f.close()
    net.train()
    ncount = 0
    sum = 0
    for batch_index, (images, labels) in enumerate(train_loader):
        ncount += 1
        images = Variable(images)
        labels = Variable(labels)
        # print(labels)
        # print(images.shape)
        labels = labels.cuda()
        images = images.cuda()
        optimizer.zero_grad()
        images = images
        outputs = net(images)
        # print(outputs.shape)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(train_loader.dataset)
        ))
        writer.add_scalar('Train_loss', loss.data.item(), epoch * len(train_loader) + ncount)
        sum = sum + loss.item()
    avg_loss = sum / ncount
    fw = open("loss.txt", "a")
    fw.write(str(avg_loss))
    fw.write("\n")
    fw.close()
    return avg_loss



def eval_training(net):  # 用来计算平均损失和平均准确率
    net.eval()
    f1 = open("accuracy.txt", "a")
    test_loss = 0.0  # cost function error
    correct = 0.0
    for (images, labels) in test_loader:
        images = Variable(images)
        labels = Variable(labels)
        # print(labels)
        # print(type(labels))
        images = images.cuda()
        labels = labels.cuda()
        images = images
        outputs = net(images)
        loss = loss_function(outputs, labels)  # 损失函数
        test_loss += loss.item()  # 张量里的元素值
        _, preds = outputs.max(1)  # 最大值的index
        correct += preds.eq(labels).sum()  # 统计相等的个数

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset)
    ))

    f1.write('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset)
    ))
    print()
    f1.close()

    return correct.float() / len(test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data path')
    parser.add_argument('--w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--ce', type=bool, default=False, help='if continue train')
    parser.add_argument('--weights', type=str, default="", help='the weights file you want to test')
    parser.add_argument('--epoch', type=int, default=100, help='epoch of train')

    args = parser.parse_args()

    writer = SummaryWriter('./Result')

    print("begin to train mobilenetV1")

    if(args.ce == True):
        model = MobileNetV1()
        model.load_state_dict(torch.load(args.weights))
        print("load")
    else:
        model = MobileNetV1()
        print("create net")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root=args.data, train=True,  download=False, transform=transform_train)
    testset  = torchvision.datasets.CIFAR10(root=args.data, train=False, download=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    model = model.cuda()

    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    iter_per_epoch = len(train_loader)
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())).replace(':', '-')
    checkpoint_path = os.path.join(CHECKPOINT_PATH, "mobilenetV1", t)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(0, args.epoch):
        avg_loss = train(epoch, model)  # 训练
        acc = eval_training(model)
        
        if  best_acc < acc:
            torch.save(model.state_dict(), checkpoint_path.format(net="mobilenetV1", epoch=epoch, type='best'))
            best_acc = acc
            print(checkpoint_path.format(net="mobilenetV1", epoch=epoch, type='regular'))
            continue

        if not epoch % SAVE_EPOCH:
            print("保存")
            torch.save(model.state_dict(), checkpoint_path.format(net="mobilenetV1", epoch=epoch, type='regular'))
            print(checkpoint_path.format(net="mobilenetV1", epoch=epoch, type='regular'))