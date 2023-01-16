import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import math

PATH = './model/'

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
DEVICE = torch.device("cpu")

BATCH = 64

IMAGESIZE = 28


DATA = torchvision.datasets.MNIST(
    download=False, root="./data", train=True, transform=transforms.Compose([transforms.ToTensor()]))

Dataloader = data.DataLoader(dataset=DATA, batch_size=BATCH,
                             shuffle=True, pin_memory=True, num_workers=6, drop_last=True)


class AAE(nn.Module):
    def __init__(self):
        super(AAE, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in [1, 2]:
            self.encoder.append(nn.Sequential(
                nn.Conv2d(2**(i-1), 2**(i), 4, 2, 1, bias=False),
                nn.BatchNorm2d(2**i),
                nn.LeakyReLU()
            ))
        self.encoder.append(nn.Sequential(
            nn.Conv2d(4, 8, 7, 1, 0, bias=False),
            nn.BatchNorm2d(8),
            nn.Sigmoid()
        ))

        self.decoder.append(nn.Sequential(
            nn.ConvTranspose2d(8, 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU()
        ))
        for i in [2, 1]:
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(2**(i), 2**(i-1), 4, 2, 1, bias=False),
                nn.BatchNorm2d(2**(i-1)),
                nn.Sigmoid()
            ))

    def decode(self, value):
        output = value

        for decoder_layer in self.decoder:
            output = decoder_layer(output)
        return output

    def encode(self, value):
        output = value
        for encoder_layer in self.encoder:
            output = encoder_layer(output)
        return output

    def forward(self, value):
        z = self.encode(value)
        output = self.decode(z)
        return output


class CriticDecoder(nn.Module):
    def __init__(self):
        super(CriticEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Conv2d(1, 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(),
            nn.Conv2d(2, 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
        ))
        self.layers.append(nn.Sequential(
            nn.Linear(8, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1)
        ))

    def forward(self, value, batch):
        output = self.layers[0](value).reshape(batch, -1)
        return self.layers[1](output)


class CriticEncoder(nn.Module):
    def __init__(self):
        super(CriticEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Linear(9, 9),
            nn.LeakyReLU()
        ))
        self.layers.append(nn.Sequential(
            nn.Linear(9, 1)
        ))

    def forward(self, value, label, batch):
        output = value

        output = torch.cat(
            (output.reshape(batch, -1), label.reshape(batch, 1)), 1)

        for layer in self.layers:
            output = layer(output)
        return output


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def train(epoch):
    aae = AAE()
    aae.train()
    aae.apply(initialize_weights)
    aae.to(device=DEVICE)
    print(aae)

    critic = CriticEncoder()
    critic.train()
    critic.apply(initialize_weights)
    critic.to(device=DEVICE)

    optim_critic = optim.RMSprop(critic.parameters(), lr=0.00005)
    optim_aae = optim.AdamW(aae.parameters(), lr=0.00005)
    loss_function = nn.BCELoss()
    writer = SummaryWriter("runs/GAN/test")

    loss_encoder_arr = []
    loss_decoder_arr = []
    loss_critic_arr = []
    for e in range(epoch):
        print(f"epoch: {e+1}")
        for iter, (target, label) in enumerate(tqdm(Dataloader)):
            target = target.to(device=DEVICE)
            label = label.to(device=DEVICE)

            # train critic
            for _ in range(6):
                random = torch.rand((BATCH, 8), device=DEVICE)

                critic_rand = critic(random, label, BATCH)

                hidden_critic = aae.encode(target).reshape(BATCH, -1)

                critic_hidden = critic(hidden_critic, label, BATCH).reshape(-1)

                # wasserstein loss
                loss_critic = -(torch.mean(critic_rand) -
                                torch.mean(critic_hidden))

                loss_critic_arr.append(loss_critic.item())
                for param in critic.parameters():
                    param.grad = None
                loss_critic.backward(retain_graph=True)
                optim_critic.step()

                # weight clipping
                for p in critic.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # train encoder
            encoded = aae.encode(target).reshape(BATCH, -1)
            critic_output = critic(encoded, label, BATCH)

            loss_encoder = -torch.mean(critic_output)
            loss_encoder_arr.append(loss_encoder.item())

            for param in aae.parameters():
                param.grad = None
            loss_encoder.backward()
            optim_aae.step()

            # train AAE
            decoded = aae(target)
            loss_decoder = loss_function(decoded, target)
            loss_decoder_arr.append(loss_decoder.item())

            for param in aae.parameters():
                param.grad = None
            loss_decoder.backward()
            optim_aae.step()

            global_step = e*len(Dataloader)+iter
            logScalarToTensorboard(
                lc=loss_critic_arr, ld=loss_decoder_arr, le=loss_encoder_arr, step=global_step, writer=writer)

            if global_step % 250 == 0 or global_step == 1:
                logImageToTensorboard(aae=aae,
                                      step=global_step, target=target, writer=writer)
                loss_encoder_arr = []
                loss_decoder_arr = []
                loss_critic_arr = []

        print("----------------\n\rsaving...")
        torch.save(aae, PATH + "net.pth")
        torch.save(critic, PATH + "Cnet.pth")
        writer.flush()
        print("saved\n\r----------------")
    writer.flush()
    writer.close()


def logScalarToTensorboard(lc, ld, le, writer, step):
    with torch.no_grad():
        writer.add_scalars('current_run', {
            'loss_Critic': torch.mean(torch.tensor(lc)).item(),
            'loss_Decoder': torch.mean(torch.tensor(ld)).item(),
            'loss_Encoder': torch.mean(torch.tensor(le)).item()
        }, global_step=step)


def logImageToTensorboard(writer, step, aae, target):
    with torch.no_grad():
        noise = aae.decode(torch.rand((BATCH, 8, 1, 1), device=DEVICE))

        noise_grid = torchvision.utils.make_grid(
            noise[:BATCH], normalize=True
        )

        writer.add_image("NoiseImage", noise_grid, global_step=step)

        test = aae(target)

        test_grid = torchvision.utils.make_grid(
            test[:BATCH], normalize=True
        )

        writer.add_image("TestImage", test_grid, global_step=step)

        real_grid = torchvision.utils.make_grid(
            target[:BATCH], normalize=True
        )

        writer.add_image("RealImage", real_grid, global_step=step)


def start():
    train(200)


if __name__ == "__main__":
    start()

    if torch.cuda.is_available():
        print("---\r\nGPU MODE\r\n---")
    else:
        print("---\r\nCPU MODE\r\n---")
