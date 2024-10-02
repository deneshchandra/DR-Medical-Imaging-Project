import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
from sklearn.utils import class_weight
import os

np.random.seed(42)
torch.manual_seed(42)

class CNNModel(nn.Module):
    def __init__(self, nb_filters, kernel_size, nb_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=nb_filters, kernel_size=kernel_size, stride=4)
        self.conv2 = nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=nb_filters, out_channels=64, kernel_size=(16, 16))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 9 * 9, 128)  # Update the input size based on your architecture
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, nb_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv4(x))
        x = self.flatten(x)
        x = nn.functional.sigmoid(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return nn.functional.softmax(x, dim=1)


def split_data(X, y, test_data_size):
    return train_test_split(X, y, test_size=test_data_size, random_state=42)


def reshape_data(arr, img_rows, img_cols, channels):
    return arr.reshape(arr.shape[0], channels, img_rows, img_cols)  # Note: PyTorch uses (C, H, W)


def train_model(model, X_train, y_train, batch_size, nb_epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Create a dataset and data loader
    train_dataset = data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(nb_epoch):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {loss.item():.4f}')


def save_model(model, score, model_name):
    if score >= 0.75:
        print("Saving Model")
        torch.save(model.state_dict(), f"../models/{model_name}_recall_{round(score, 4)}.pth")
    else:
        print("Model Not Saved. Score: ", score)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # Specify parameters before model is run.
    batch_size = 1000
    nb_classes = 5
    nb_epoch = 30

    img_rows, img_cols = 256, 256
    channels = 3
    nb_filters = 32
    kernel_size = (8, 8)

    # Import data
    labels = pd.read_csv("../labels/trainLabels_master_256_v2.csv")
    X = np.load("../data/X_train_256_v2.npy")
    y = np.array(labels['level'])

    # Class Weights (for imbalanced classes)
    print("Computing Class Weights")
    weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

    print("Splitting data into test/train datasets")
    X_train, X_test, y_train, y_test = split_data(X, y, 0.2)

    print("Reshaping Data")
    X_train = reshape_data(X_train, img_rows, img_cols, channels)
    X_test = reshape_data(X_test, img_rows, img_cols, channels)

    print("X_train Shape: ", X_train.shape)
    print("X_test Shape: ", X_test.shape)

    print("Normalizing Data")
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    y_train = np.eye(nb_classes)[y_train]  # Convert labels to one-hot encoding
    y_test = np.eye(nb_classes)[y_test]
    print("y_train Shape: ", y_train.shape)
    print("y_test Shape: ", y_test.shape)

    print("Training Model")
    model = CNNModel(nb_filters, kernel_size, nb_classes)
    train_model(model, X_train, y_train, batch_size, nb_epoch)

    print("Predicting")
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, dtype=torch.float32))

    y_pred = torch.argmax(y_pred, dim=1).numpy()
    y_test_labels = np.argmax(y_test, axis=1)

    precision = precision_score(y_test_labels, y_pred, average='weighted')
    recall = recall_score(y_test_labels, y_pred, average='weighted')

    print("Precision: ", precision)
    print("Recall: ", recall)

    save_model(model=model, score=recall, model_name="DR_Two_Classes")

    print("Completed")
