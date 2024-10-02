import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import kappa

np.random.seed(42)

# CNN Model Definition
class CNNModel(nn.Module):
    def __init__(self, nb_classes, nb_filters):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, nb_filters, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(nb_filters, nb_filters, kernel_size=4, stride=1, padding=1)
        self.conv3 = nn.Conv2d(nb_filters, nb_filters, kernel_size=4, stride=1, padding=1)
        self.pool = nn.MaxPool2d(8, 8)
        self.fc1 = nn.Linear(nb_filters * 30 * 30, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, nb_classes)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class EyeNet:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_data_size = None
        self.weights = None
        self.model = None
        self.nb_classes = None
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3

    def split_data(self, y_file_path, X, test_data_size=0.2):
        labels = pd.read_csv(y_file_path)
        self.X = np.load(X)
        self.y = np.array(labels['level'])
        self.weights = class_weight.compute_class_weight('balanced', np.unique(self.y), self.y)
        self.test_data_size = test_data_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.test_data_size,
                                                                                random_state=42)

    def reshape_data(self, img_rows, img_cols, channels, nb_classes):
        self.nb_classes = nb_classes
        self.X_train = self.X_train.reshape(self.X_train.shape[0], channels, img_rows, img_cols).astype("float32") / 255
        self.X_test = self.X_test.reshape(self.X_test.shape[0], channels, img_rows, img_cols).astype("float32") / 255

        self.y_train = torch.tensor(self.y_train).long()
        self.y_test = torch.tensor(self.y_test).long()

        print("X_train Shape: ", self.X_train.shape)
        print("X_test Shape: ", self.X_test.shape)
        print("y_train Shape: ", self.y_train.shape)
        print("y_test Shape: ", self.y_test.shape)

    def cnn_model(self, nb_filters, batch_size, nb_epoch):
        # Initialize CNN model
        self.model = CNNModel(nb_classes=self.nb_classes, nb_filters=nb_filters).cuda()

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Data loaders
        train_dataset = TensorDataset(torch.tensor(self.X_train).float().cuda(), self.y_train.cuda())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(nb_epoch):
            self.model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{nb_epoch}], Step [{i + 1}], Loss: {loss.item():.4f}')

        return self.model

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(self.X_test).float().cuda()
            outputs = self.model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()

            precision = precision_score(self.y_test.cpu().numpy(), predicted, average="micro")
            recall = recall_score(self.y_test.cpu().numpy(), predicted, average="micro")
            f1 = f1_score(self.y_test.cpu().numpy(), predicted, average="micro")
            cohen_kappa = cohen_kappa_score(self.y_test.cpu().numpy(), predicted)
            quad_kappa = kappa(self.y_test.cpu().numpy(), predicted, weights='quadratic')

            return precision, recall, f1, cohen_kappa, quad_kappa

    def save_model(self, score, model_name):
        if score >= 0.75:
            print("Saving Model")
            torch.save(self.model.state_dict(), "../models/" + model_name + "_recall_" + str(round(score, 4)) + ".pt")
        else:
            print("Model Not Saved. Score: ", score)


if __name__ == '__main__':
    cnn = EyeNet()
    cnn.split_data(y_file_path="../labels/trainLabels_master_256_v2.csv", X="../data/X_train_256_v2.npy")
    cnn.reshape_data(img_rows=256, img_cols=256, channels=3, nb_classes=5)
    model = cnn.cnn_model(nb_filters=32, batch_size=512, nb_epoch=50)
    precision, recall, f1, cohen_kappa, quad_kappa = cnn.predict()
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("Cohen Kappa Score", cohen_kappa)
    print("Quadratic Kappa: ", quad_kappa)
    cnn.save_model
