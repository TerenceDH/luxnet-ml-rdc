"""Original MResUNet architecture; state-dict names are preserved."""
import torch
import torch.nn as nn
import torch.nn.functional as F
COND_HIDDEN_DIM = FUSION_DIM = 64
USE_DROPOUT = True
DROPOUT_RATE = 0.1
INPUT_NC, OUTPUT_NC, BASE_CH, N_DOWN, N_BLOCKS = 3, 1, 16, 2, 3
def get_norm(c):
    return nn.InstanceNorm2d(c, affine=False)


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_dropout=False, dropout_rate=0.1):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            get_norm(dim),
            nn.LeakyReLU(0.2, True)
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(dropout_rate))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            get_norm(dim),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class CondEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, out_dim),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.net(x)


class DualHeadUNet(nn.Module):
    """
    图像头: A_float [3,128,128]
    向量头: cond_vec [D]
    融合方式: bottleneck FiLM
    """
    def __init__(self, in_nc, out_nc, cond_dim, base_ch=32, n_down=3, n_blocks=3):
        super().__init__()

        self.cond_encoder = CondEncoder(cond_dim, hidden_dim=COND_HIDDEN_DIM, out_dim=FUSION_DIM)

        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_nc, base_ch, 7),
            get_norm(base_ch),
            nn.LeakyReLU(0.2, True)
        )

        self.encoder_features = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()

        curr_ch = base_ch
        enc_chs = []
        for _ in range(n_down):
            self.encoder_features.append(
                ResnetBlock(curr_ch, use_dropout=USE_DROPOUT, dropout_rate=DROPOUT_RATE)
            )
            enc_chs.append(curr_ch)

            next_ch = curr_ch * 2
            self.encoder_downs.append(nn.Sequential(
                nn.Conv2d(curr_ch, next_ch, 3, stride=2, padding=1),
                get_norm(next_ch),
                nn.LeakyReLU(0.2, True)
            ))
            curr_ch = next_ch

        self.bottleneck = nn.Sequential(*[
            ResnetBlock(curr_ch, use_dropout=USE_DROPOUT, dropout_rate=DROPOUT_RATE)
            for _ in range(n_blocks)
        ])

        self.to_gamma_beta = nn.Linear(FUSION_DIM, curr_ch * 2)
        # self.to_bias = nn.Linear(FUSION_DIM, curr_ch)

        self.decoders = nn.ModuleList()
        for i in range(n_down):
            skip_ch = enc_chs[-(i + 1)]
            up_ch = curr_ch // 2
            self.decoders.append(nn.ModuleDict({
                "up": nn.Sequential(
                    # nn.Upsample(scale_factor=2.0, mode='nearest'),
                    nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(curr_ch, up_ch, 3),
                    get_norm(up_ch),
                    nn.LeakyReLU(0.2, True)
                ),
                "conv": nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(up_ch + skip_ch, up_ch, 3),
                    get_norm(up_ch),
                    nn.LeakyReLU(0.2, True),
                    ResnetBlock(up_ch, use_dropout=USE_DROPOUT, dropout_rate=DROPOUT_RATE)
                )
            }))
            curr_ch = up_ch

        self.tail = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(curr_ch, out_nc, 7),
            nn.Softplus(beta=1, threshold=15)
        )

    def forward(self, x, cond):
        cond_feat = self.cond_encoder(cond)

        x = self.head(x)
        skips = []

        for feat, down in zip(self.encoder_features, self.encoder_downs):
            x = feat(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        gamma_beta = self.to_gamma_beta(cond_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        # bias = self.to_bias(cond_feat).unsqueeze(-1).unsqueeze(-1)

        # x = (1.0 + gamma) * x + beta + bias
        x = (1.0 + gamma) * x + beta
        
        for i, dec in enumerate(self.decoders):
            x = dec["up"](x)
            skip = skips[-(i + 1)]
            if x.shape[-2:] != skip.shape[-2:]:
                # x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec["conv"](x)

        return self.tail(x)


def load_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = DualHeadUNet(3, 1, 11, base_ch=16, n_down=2, n_blocks=3).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model, checkpoint['cond_mean'].numpy(), checkpoint['cond_std'].numpy()
