import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

np.random.seed(42)


def split_data(X, y, test_data_size):
    return train_test_split(X, y, test_size=test_data_size, random_state=42)

def reshape_data(arr, img_rows, img_cols, channels):
    return arr.reshape(arr.shape[0], channels, img_rows, img_cols)  # PyTorch expects (N, C, H, W)


class CNNModel(nn.Module):
    def __init__(self, nb_filters, kernel_size, nb_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=nb_filters, kernel_size=kernel_size, padding='valid')
        self.conv2 = nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(nb_filters * 29 * 29, 128)  # Adjust input size based on pooling
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, nb_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.sigmoid(self.fc1(x))  # Using sigmoid as per the original architecture
        x = self.dropout(x)
        x = torch.softmax(self.fc2(x), dim=1)
        return x


def save_model(model, score, model_name):
    if score >= 0.75:
        print("Saving Model")
        torch.save(model.state_dict(), f"../models/{model_name}_recall_{round(score, 4)}.pth")
    else:
        print("Model Not Saved.  Score: ", score)


if __name__ == '__main__':
    batch_size = 512
    nb_classes = 2
    nb_epoch = 30

    img_rows, img_cols = 256, 256
    channels = 3
    nb_filters = 32
    kernel_size = (8, 8)

    labels = pd.read_csv("../labels/trainLabels_master_256_v2.csv")
    X = np.load("../data/X_train_256_v2.npy")
    y = np.array([1 if lab >= 1 else 0 for lab in labels['level']])

    print("Splitting data into test/ train datasets")
    X_train, X_test, y_train, y_test = split_data(X, y, 0.2)

    print("Reshaping Data")
    X_train = reshape_data(X_train, img_rows, img_cols, channels)
    X_test = reshape_data(X_test, img_rows, img_cols, channels)

    print("X_train Shape: ", X_train.shape)
    print("X_test Shape: ", X_test.shape)

    print("Normalizing Data")
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    print("Training Model")
    model = CNNModel(nb_filters, kernel_size, nb_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss includes softmax
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(nb_epoch):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {loss.item():.4f}')

    print("Predicting")
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

    score = criterion(y_pred, y_test).item()
    print('Test score:', score)

    y_test = y_test.numpy()
    y_pred = torch.argmax(y_pred, dim=1).numpy()

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Precision: ", precision)
    print("Recall: ", recall)

    save_model(model=model, score=recall, model_name="DR_Two_Classes")
    print("Completed")
