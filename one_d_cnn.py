from torch import nn

# Model architecture
class MyModel(nn.Module):
    def __init__(self, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.lay1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # reusable

        self.lay2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=1),
            nn.ReLU(inplace=True)
        )

        self.lay3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=1),
            nn.ReLU(inplace=True)
        )

        self.lay4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=11, padding=1),
            nn.ReLU(inplace=True)
        )

        self.lay5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.lin1 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=0.5)
        )

        self.lin2 = nn.Sequential(
            nn.Linear(1024, seq_length)
        )

    # Function implementing the forward pass
    def forward(self, x):

        out = self.lay1(x)
        out = self.pool(out)
        out = self.lay2(out)
        out = self.pool(out)
        out = self.lay3(out)
        out = self.pool(out)
        out = self.lay4(out)
        out = self.pool(out)
        out = self.lay5(out)
        out = self.pool(out)
        out = self.lin1(out)
        out = self.lin2(out)
        return out
