import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        c = torch.zeros(B, self.hidden_dim, H, W, device=x.device)

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)

        return h  # final hidden state → (B, hidden_dim, H, W)

class SimpleCNNHead(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.net(x)

        # enforce h >= 0 on channel 0
        x = torch.cat([torch.relu(x[:, 0:1]), x[:, 1:]], dim=1)
        return x


class FloodConvLSTMModel(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()

        # ConvLSTM: input_dim = 5 (h,u,v,bathy,n)
        self.convlstm = ConvLSTM(input_dim=5, hidden_dim=hidden_dim)

        # CNN head: receives ConvLSTM_output + 5 forcing channels
        self.cnn_head = SimpleCNNHead(in_channels=hidden_dim + 5,
                                      out_channels=3)

    def forward(self, X):
        # X shape = (B,19,H,W)
        h = X[:, 0:4]      # (B,4,H,W)
        u = X[:, 4:8]      # (B,4,H,W)
        v = X[:, 8:12]     # (B,4,H,W)

        rainfall  = X[:, 12:13]
        wind_u    = X[:, 13:14]
        wind_v    = X[:, 14:15]
        pressure  = X[:, 15:16]
        discharge = X[:, 16:17]

        bathy = X[:, 17:18]  # (B,1,H,W)
        n     = X[:, 18:19]

        # ------------------------------
        # 1) Prepare ConvLSTM input
        # ------------------------------
        B, _, H, W = h.shape

        dyn_list = []
        for t in range(4):
            dyn = torch.cat([
                h[:, t:t+1],
                u[:, t:t+1],
                v[:, t:t+1],
                bathy,
                n
            ], dim=1)  # (B,5,H,W)
            dyn_list.append(dyn)

        convlstm_input = torch.stack(dyn_list, dim=1)  # (B,4,5,H,W)

        # ------------------------------
        # 2) ConvLSTM forward
        # ------------------------------
        H_t = self.convlstm(convlstm_input)  # (B,hidden_dim,H,W)

        # ------------------------------
        # 3) prepare CNN head input
        # ------------------------------
        forcing = torch.cat([
            rainfall, wind_u, wind_v, pressure, discharge
        ], dim=1)  # (B,5,H,W)

        cnn_input = torch.cat([H_t, forcing], dim=1)  # (B,hidden_dim+5,H,W)

        # ------------------------------
        # 4) final CNN output
        # ------------------------------
        out = self.cnn_head(cnn_input)  # (B,3,H,W)
        return out
