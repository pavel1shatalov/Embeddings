from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool1d, Module, BatchNorm2d, Dropout, Sigmoid, Flatten


class CNN1d(Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.conv1_input = Conv1d(1, 1, kernel_size=3, padding=1)
        self.relu = ReLU()
        self.conv1 = Conv1d(1, 16, kernel_size=3)
        self.conv2  = Conv1d(16, 16, kernel_size=3, padding=1)
        self.pool1 = MaxPool1d(kernel_size=3)
        self.flatten = Flatten()
        self.linear1 = Linear(16,10)
        self.linear2 = Linear(10, 10)
        self.linear3 = Linear(10, 1)


    def forward(self, x):
        x = self.relu(self.conv1_input(x))
        #x = self.relu(self.conv1_input(x))
        x = self.relu(self.conv1(x))
        #x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        #x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x