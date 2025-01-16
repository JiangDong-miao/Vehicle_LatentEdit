import torch
import torch.nn as nn


class PositionalEncodingXYZ(nn.Module):
    def __init__(self, hidden_dim):
        super(PositionalEncodingXYZ, self).__init__()
        self.hidden_dim = hidden_dim
        self.n2pi = (
            torch.pow(2, torch.arange(int(hidden_dim), device="cuda:0")).float()
            * torch.pi
        )

    def forward(self, x):
        # x: (batch_size, num_points, 3)
        pos_x = x[:, 0]
        pos_y = x[:, 1]
        pos_z = x[:, 2]

        pos_x = pos_x.unsqueeze(-1).repeat(1, int(self.hidden_dim)) * self.n2pi
        pos_y = pos_y.unsqueeze(-1).repeat(1, int(self.hidden_dim)) * self.n2pi
        pos_z = pos_z.unsqueeze(-1).repeat(1, int(self.hidden_dim)) * self.n2pi

        pos_x = torch.cat([torch.sin(pos_x), torch.cos(pos_x)], dim=-1)
        pos_y = torch.cat([torch.sin(pos_y), torch.cos(pos_y)], dim=-1)
        pos_z = torch.cat([torch.sin(pos_z), torch.cos(pos_z)], dim=-1)

        return torch.cat([pos_x, pos_y, pos_z], dim=-1)


class DeepSDF(nn.Module):
    def __init__(
        self, hidden_dim=512, xyz_pos_enc_dim=3, latent_code_dim=128, dropout_prob=0.001
    ):
        super(DeepSDF, self).__init__()

        self.positional_encoding = PositionalEncodingXYZ(hidden_dim=xyz_pos_enc_dim)

        self.layer0 = nn.utils.weight_norm(
            nn.Linear(
                latent_code_dim + 6 * xyz_pos_enc_dim,
                hidden_dim,
                dtype=torch.float32,
            )
        )
        self.layer1 = nn.utils.weight_norm(
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        )
        self.layer2 = nn.utils.weight_norm(
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        )
        self.layer3 = nn.utils.weight_norm(  # skip connection
            nn.Linear(
                hidden_dim,
                hidden_dim - (6 * xyz_pos_enc_dim + latent_code_dim),
                dtype=torch.float32,
            )
        )
        self.layer4 = nn.utils.weight_norm(
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        )
        self.layer5 = nn.utils.weight_norm(
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        )
        self.layer6 = nn.utils.weight_norm(
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        )
        self.layer7 = nn.utils.weight_norm(  # skip connection
            nn.Linear(
                hidden_dim,
                hidden_dim - (6 * xyz_pos_enc_dim + latent_code_dim),
                dtype=torch.float32,
            )
        )
        self.layer8 = nn.utils.weight_norm(
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        )
        self.layer9 = nn.utils.weight_norm(
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        )
        self.layer10 = nn.utils.weight_norm(
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        )
        self.layer11 = nn.utils.weight_norm(  # skip connection
            nn.Linear(
                hidden_dim,
                hidden_dim - (6 * xyz_pos_enc_dim + latent_code_dim),
                dtype=torch.float32,
            )
        )
        self.layer12 = nn.utils.weight_norm(
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        )
        self.layer13 = nn.utils.weight_norm(
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        )
        self.layer14 = nn.utils.weight_norm(
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        )
        self.output = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, latent, xyz):
        latent = latent.repeat(xyz.shape[0], 1)
        xyz = self.positional_encoding(xyz)
        input = torch.cat([latent, xyz], dim=-1)

        
        # layer-norm無し
        out0 = self.dropout(torch.relu(self.layer0(input)))
        out1 = self.dropout(torch.relu(self.layer1(out0)))
        out2 = self.dropout(torch.relu(self.layer2(out1)))
        out3 = self.dropout(torch.relu(self.layer3(out2)))
        out4 = self.dropout(torch.relu(self.layer4(torch.cat([out3, input], dim=-1))))
        out5 = self.dropout(torch.relu(self.layer5(out4)))
        out6 = self.dropout(torch.relu(self.layer6(out5)))
        out7 = self.dropout(torch.relu(self.layer7(out6)))
        out8 = self.dropout(torch.relu(self.layer8(torch.cat([out7, input], dim=-1))))
        out9 = self.dropout(torch.relu(self.layer9(out8)))
        out10 = self.dropout(torch.relu(self.layer10(out9)))
        out11 = self.dropout(torch.relu(self.layer11(out10)))
        out12 = self.dropout(
            torch.relu(self.layer12(torch.cat([out11, input], dim=-1)))
        )
        out13 = self.dropout(torch.relu(self.layer13(out12)))
        out14 = self.dropout(torch.relu(self.layer14(out13)))
        out = nn.functional.tanh(self.output(out14))
        return out
