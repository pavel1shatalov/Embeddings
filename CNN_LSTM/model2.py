from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool1d, Module, BatchNorm2d, Dropout, Sigmoid, Flatten


class CNN1d(Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.conv1_input = Conv1d(self.num_channels, self.num_channels, kernel_size=3, padding=1)
        self.relu = ReLU()
        self.conv1 = Conv1d(self.num_channels, 32, kernel_size=3)
        self.conv2 = Conv1d(32, 32, kernel_size=3)
        self.pool1 = MaxPool1d(kernel_size=2)
        self.conv3 = Conv1d(32, 16, kernel_size=3)
        self.flatten = Flatten()
        self.linear1 = Linear(16,100)
        self.linear2 = Linear(100, 50)
        self.linear3 = Linear(50, 1)


    def forward(self, x):
        x = x.transpose(1,2)
        x = self.relu(self.conv1_input(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x