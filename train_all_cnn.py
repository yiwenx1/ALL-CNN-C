import hw2.all_cnn
import hw2.preprocessing as P
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
import torch.utils.data as Data
import os
from torchvision.datasets import CIFAR10
"""
Refer to handout for details.
- Build scripts to train your model
- Submit your code to Autolab
"""


def write_results(predictions, output_file='predictions.txt'):
    """
    Write predictions to file for submission.
    File should be:
        named 'predictions.txt'
        in the root of your tar file
    :param predictions: iterable of integers
    :param output_file:  path to output file.
    :return: None
    """
    with open(output_file, 'w') as f:
        for y in predictions:
            f.write("{}\n".format(y))


def train(train_loader, model, criterion, optimizer, epoch):
    running_loss = 0
    correct = 0
    total = 0
    for step, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.float().cuda()
        labels = labels.type(torch.LongTensor).cuda()
        # print("inputs {}, labels {}".format(inputs, labels))
        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        print("predictions: {}, labels: {}".format(predicted, labels))
        
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        print("loss at every step: ", loss.item())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if step % 10 == 9:
            print("step {}, loss {}, accuracy {}".format(step, running_loss / total, correct / total))
            running_loss = 0
            correct = 0
            total = 0
    save_model(epoch, model, optimizer, loss, step, "/home/ubuntu/ALL-CNN-C/weights/")


def save_model(epoch, model, optimizer, loss, step, save_path):
    filename = save_path + str(epoch) + '-' + str(step) + '-' + "%.6f" % loss.item() + '.pth'
    print('Save model at Train Epoch: {} [Step: {}\tLoss: {:.12f}]'.format(
        epoch, step, loss.item()))
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, filename)


def load_model(epoch, step, loss, model, optimizer, save_path):
    filename = save_path + str(epoch) + '-' + str(step) + '-' + str(loss) + '.pth'
    if os.path.isfile(filename):
        print("######### loading weights ##########")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss = checkpoint['loss']
        print('########## loading weights done ##########')
        return model, optimizer, start_epoch, loss
    else:
        print("no such file: ", filename)
    

def test(test_loader, model, criterion):
    predictions = []
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(test_loader):
            print(step)
            inputs = inputs.float().cuda()
            labels = labels.type(torch.LongTensor).cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.cpu().numpy())
            print(predicted)
    predictions = np.array(predictions)
    predictions = predictions.flatten()
    return predictions


class ALL_CNN_C(nn.Module):
    def __init__(self):
        super(ALL_CNN_C, self).__init__()
        self.model = hw2.all_cnn.all_cnn_module()
    
    def forward(self, x):
        x = self.model(x)
        return x

class CostomDataset(Dataset):
    def __init__(self, X, Y, test_X, mode):
        if mode == "train":
            self.X = X
            self.Y = Y
        else:
            self.test_X = test_X
        self.mode = mode
        # self.X, self.test_X = P.cifar_10_preprocess(self.X, self.test_X)
    
    def __len__(self):
        if self.mode == "train":
            return len(self.X)
        else:
            return len(self.test_X)
    
    def __getitem__(self, index):
        if self.mode == "train":
            return self.X[index].reshape((3, 32, 32)), self.Y[index]
        else:
            return self.test_X[index].reshape((3, 32, 32)), -1


def main(nepochs):
    X = np.load("/home/ubuntu/ALL-CNN-C/dataset/train_feats.npy")
    print("loading X done")
    Y = np.load("/home/ubuntu/ALL-CNN-C/dataset/train_labels.npy")
    print("loading Y done")
    test_X = np.load("/home/ubuntu/ALL-CNN-C/dataset/test_feats.npy") 
    print("loading test_X done")
    X, test_X = P.cifar_10_preprocess(X, test_X)
    print("preprocessing done")

    train_data = CostomDataset(X, Y, test_X, "train")
    test_data = CostomDataset(X, Y, test_X, "test")

    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = ALL_CNN_C()
    print("model: ", model)
    if torch.cuda.is_available():
        print("model in GPU mode")
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    start_epoch = 0
    model, optimizer, start_epoch, loss = load_model(
        load_epoch, 
        load_step, 
        load_loss, 
        model,
        optimizer, 
        "/home/ubuntu/ALL-CNN-C/weights/")

    for epoch in range(start_epoch, nepochs):
        model.train()
        print("########## epoch {} ##########".format(epoch))
        train(train_loader, model, criterion, optimizer, epoch)
    print("########## Finished Training ##########")
    model.eval()
    predictions = test(test_loader, model, criterion) 
    print("########## Finished Testing ##########")
    write_results(predictions)


if __name__ == '__main__':
    nepochs = 20
    batch_size = 50
    learning_rate = 0.01
    load_epoch = 19
    load_step = 999
    load_loss = 0.284484
    main(nepochs)
