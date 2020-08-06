import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
from model2 import resnet18
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='/core7/wanghaoran/cifar10', train=True,
                                            download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # net = Net().to(device)
    net = torch.nn.DataParallel(resnet18(num_classes=10).to(device))
    net.cuda()

    plt.figure()  # 定义一个图像窗口

    x = list()
    y = list()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    t = time.clock()
    ti = time.clock()
    for epoch in range(200):  # loop over the dataset multiple times
        # print(optimizer)
        net.train()
        # if epoch == 20:
        #     optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
        # if epoch == 40:
        #     optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        if epoch == 40:
            optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        if epoch == 80:
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        if epoch == 110:
            optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
        if epoch == 140:
            optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs, labels = Variable(inputs), Variable(
                labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            x.append(1)
            y.append(1)
            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                print(running_loss / 200)

                x[epoch] = epoch
                y[epoch] = running_loss/200
                # plt.plot(epoch, running_loss/200)
                plt.scatter(x[epoch], y[epoch])

                running_loss = 0.0

        s = time.clock() - t
        t = time.clock()
        print("耗时为：%s" % s)
    print('Finished Training')
    plt.show()
    s = time.clock() - ti
    print("总耗时为：%s" % s)

    testset = torchvision.datasets.CIFAR10(root='/core7/wanghaoran/cifar10', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=4)
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            net.eval()
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    PATH = 'model.pth'
    torch.save(net, 'model.pth')