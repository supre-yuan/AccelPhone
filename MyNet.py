import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms



class MyDataset(Dataset):

    def __init__(self, root):
        global deviceBook, wordBook
        deviceBook = {"mi": 0, "s9": 1, "acc": 2, "oppo": 3, "iphone": 4}
        wordBook = {"pswd": 0, "key": 1, "secret": 2, "code": 3,
                    "account": 4, "salary": 5, "encoder": 6, "bank": 7,
                    "number": 8, "word": 9}
        reader = pd.read_excel(root)
        self.image_addrs = reader["addr"].values
        # =====one-hot encoding==========
        temp_imag_wordlabels = reader.iloc[:, 1].values
        temp_imag_devicelabels = reader.iloc[:, 2].values

        self.imag_wordlabels = []
        self.imag_devicelabels = []

        for word in temp_imag_wordlabels:

            if word != "nothing":
                t = wordBook[word]
                self.imag_wordlabels.append(t)
            else:
                self.imag_wordlabels.append(np.nan)
        for device in temp_imag_devicelabels:
            self.imag_devicelabels.append(deviceBook[device])

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_addr = self.image_addrs[index]
        wordLabelName = self.imag_wordlabels[index]
        deviceLabelName = self.imag_devicelabels[index]
        image = Image.open(img_addr)
        image = self.transform(image)

        return image, \
               wordLabelName, \
               deviceLabelName

    def __len__(self):
        return len(self.image_addrs)


class CNN(nn.Module):
    '''
    Extracts features from RGB images
    '''

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                             out_channels=32,
                                             kernel_size=3,
                                             stride=1,
                                             padding=0),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3), nn.BatchNorm2d(32), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.conv3 = nn.Sequential(nn.Conv2d(32, 32, 3), nn.BatchNorm2d(32), nn.ReLU(),
                                   nn.MaxPool2d(2, 2), nn.Flatten())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class WordRecognizer(nn.Module):
    '''
    Identifies words with extracted feature vectors from outputs of CNN
    '''

    def __init__(self):
        super(WordRecognizer, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(1152, 256),
                                 nn.BatchNorm1d(256), nn.Softplus())

        self.fc2 = nn.Sequential(nn.Linear(256, 10), nn.Softmax())

    def forward(self, x):
        x = self.fc1(x)

        x = self.fc2(x)
        return x


class DeviceDiscriminator(nn.Module):
    '''
    Identifies devices with extracted feature vectors from outputs of CNN & WordRecognizer
    '''

    def __init__(self):
        super(DeviceDiscriminator, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(1162, 512),
                                 nn.Softplus())
        self.fc2 = nn.Sequential(nn.Linear(512, 3),
                                 nn.Softmax())

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def evaluate(model, cnn, loader, test_size):
    global TrainingDevice
    model.eval()
    cnn.eval()
    correct = 0
    total = test_size
    for j, (img_a, wordLabel_a, deviceLabel_a) in loader:
        x, y = img_a.to(TrainingDevice), wordLabel_a.to(TrainingDevice)
        with torch.no_grad():
            logits = model(cnn(x))
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total

def evaluate3(model, cnn, loader, test_size):
    global TrainingDevice
    model.eval()
    cnn.eval()
    correct = 0
    total = test_size
    for j, (img_a, wordLabel_a, deviceLabel_a) in loader:
        x, y = img_a.to(TrainingDevice), wordLabel_a.to(TrainingDevice)
        with torch.no_grad():
            logits = model(cnn(x))
            maxk = max((1, 3))
            y_resize = y.view(-1, 1)
            _, pred = logits.topk(maxk, 1, True, True)
        correct += torch.eq(pred, y_resize).sum().float().item()
    return correct / total

def evaluate5(model, cnn, loader, test_size):
    global TrainingDevice
    model.eval()
    cnn.eval()
    correct = 0
    total = test_size
    for j, (img_a, wordLabel_a, deviceLabel_a) in loader:
        x, y = img_a.to(TrainingDevice), wordLabel_a.to(TrainingDevice)
        with torch.no_grad():
            logits = model(cnn(x))
            maxk = max((1, 5))
            y_resize = y.view(-1, 1)
            _, pred = logits.topk(maxk, 1, True, True)
        correct += torch.eq(pred, y_resize).sum().float().item()
    return correct / total

def evaluate2(wr, dd, fe, loader, test_size):
    global TrainingDevice
    wr.eval()
    dd.eval()
    fe.eval()
    correct = 0
    total = test_size
    for j, (img_a, wordLabel_a, deviceLabel_a) in loader:
        x, y = Variable(img_a).cuda(), Variable(deviceLabel_a).cuda()
        with torch.no_grad():
            features = fe(x)
            wr_p = wr(features)
            C = torch.cat((features, wr_p), 1)
            logits = dd(C)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


TrainingDevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(Train_A, Vali):
    global deviceBook, wordBook
    featureExtractor = CNN().to(TrainingDevice)
    wordRecognizer = WordRecognizer().to(TrainingDevice)
    deviceDiscriminator = DeviceDiscriminator().to(TrainingDevice)


    LR = 0.01
    Epoch = 500
    BatchSize = 10

    Criterion = nn.CrossEntropyLoss()
    f_optimizer = torch.optim.Adam(featureExtractor.parameters(), lr=LR)
    w_optimizer = torch.optim.Adam(wordRecognizer.parameters(), lr=LR)
    d_optimizer = torch.optim.Adam(deviceDiscriminator.parameters(), lr=LR)

    dataset_a = MyDataset(root=Train_A)

    dataLoader_a = DataLoader(dataset_a, batch_size=BatchSize, shuffle=True)
    # ======== TODO:validation set============
    valset = MyDataset(root=Vali)
    valLoader = DataLoader(valset, batch_size=BatchSize, shuffle=True)
    valset_list = list(enumerate(valLoader))
    valset_size = valset.__len__()
    # ======== validation set============

    dataset_list_a = list(enumerate(dataLoader_a))
    dataset_list_length = len(dataLoader_a)

    global_step = 0
    best_acc = 0

    print("=====start training=====")
    for epoch in range(Epoch):
        for i in range(dataset_list_length):
            j, (imgs_a, wordLabels_a, deviceLabels) = dataset_list_a[i]
            imgs_a, wordLabels_a = Variable(imgs_a).cuda(), Variable(wordLabels_a).cuda()
            deviceLabels = Variable(deviceLabels).cuda()

            f_optimizer.zero_grad()
            w_optimizer.zero_grad()
            d_optimizer.zero_grad()
            features = featureExtractor(imgs_a)
            words_pred = wordRecognizer(features)

            features_copy = features.detach()
            words_pred_copy = words_pred.detach()

            loss_a = Criterion(words_pred, wordLabels_a)

            loss_a.backward()

            C = torch.cat((features_copy, words_pred_copy), 1)
            devices_pred = deviceDiscriminator(C)
            loss_d = Criterion(devices_pred, deviceLabels)

            loss_d.backward()

            LOSS = loss_a - loss_d

            f_optimizer.step()
            w_optimizer.step()
            d_optimizer.step()

            global_step += 1

        # ==========evaluation&visualize==========
        if epoch % 1 == 0:
            val_acc = evaluate(wordRecognizer,
                               featureExtractor,
                               valset_list, valset_size)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                print("best epoch=%d   best acc=%.5f" % (best_epoch, best_acc))

    # TODO： save model
    torch.save(featureExtractor, 'featureExtractor.pt')
    torch.save(wordRecognizer, 'wordRecognizer.pt')
    torch.save(deviceDiscriminator, 'deviceDiscriminator.pt')
    print("model saved")


def test(Test_file):
    global wordBook, deviceBook
    BatchSize = 10
    featureExtractor_eval = torch.load('featureExtractor.pt')
    print(featureExtractor_eval)
    wordRecognizer_eval = torch.load('wordRecognizer.pt')
    print(wordRecognizer_eval)
    testset = MyDataset(root=Test_file)
    testLoader = DataLoader(testset, batch_size=BatchSize, shuffle=True)
    test_size = len(testLoader.dataset)
    print("=======================testsize=======================", test_size)
    testset_list = list(enumerate(testLoader))

    rate = evaluate(wordRecognizer_eval, featureExtractor_eval, testset_list, test_size)
    rate3 = evaluate3(wordRecognizer_eval, featureExtractor_eval, testset_list, test_size)
    rate5 = evaluate5(wordRecognizer_eval, featureExtractor_eval, testset_list, test_size)
    print("testset correct rate=", rate, rate3, rate5)


if __name__ == '__main__':
    deviceBook = {"mi": 0, "s9": 1, "acc": 2, "oppo": 3, "iphone": 4}
    wordBook = {"pswd": 0, "key": 1, "secret": 2, "code": 3,
                "account": 4, "salary": 5, "encoder": 6, "bank": 7,
                "number": 8, "word": 9}
    Train = "labels_train.xlsx"
    Vali = "labels_vali.xlsx"
    train(Train, Vali)
    Test_file = "labels_test.xlsx"
    test(Test_file)

    valset = MyDataset(root=Vali)
    valLoader = DataLoader(valset, batch_size=10, shuffle=True)
    valset_list = list(enumerate(valLoader))
    valset_size = valset.__len__()
    dd_eval = torch.load('deviceDiscriminator.pt')
    cnn = torch.load('featureExtractor.pt')
    wr = torch.load('wordRecognizer.pt')

    r = evaluate2(wr, dd_eval, cnn, valset_list, valset_size)
    print(r)
