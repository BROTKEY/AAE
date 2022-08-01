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

BATCH = 8

IMAGESIZE = 28


DATA = torchvision.datasets.MNIST(
    download=False, root="./data", train=True, transform=transforms.Compose([transforms.ToTensor()]))

Dataloader = data.DataLoader(dataset=DATA, batch_size=BATCH,
                             shuffle=True, pin_memory=True, num_workers=0, drop_last=True)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in [1, 2]:
            self.layers.append(nn.Sequential(
                nn.Conv2d(2**(i-1), 2**(i), 4, 2, 1, bias=False),
                nn.BatchNorm2d(2**i),
                nn.LeakyReLU()
            ))
        self.layers.append(nn.Sequential(
            nn.Conv2d(4, 8, 7, 1, 0, bias=False),
            nn.BatchNorm2d(8),
            nn.Sigmoid()
        ))

    def forward(self, value):
        output = value
        for i, layer in enumerate(self.layers):
            output = layer(output)
        return output


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.ConvTranspose2d(9, 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU()
        ))
        for i in [2, 1]:
            self.layers.append(nn.Sequential(
                nn.ConvTranspose2d(2**(i), 2**(i-1), 4, 2, 1, bias=False),
                nn.BatchNorm2d(2**(i-1)),
                nn.LeakyReLU()
            ))

    def forward(self, value):
        output = value
        for i, layer in enumerate(self.layers):
            output = layer(output)
        return output


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Linear(9, 9),
            nn.LeakyReLU()
        ))
        self.layers.append(nn.Sequential(
            nn.Linear(9, 1)
        ))

    def forward(self, value):
        output = value
        for i, layer in enumerate(self.layers):
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
    encoder = Encoder()
    encoder.train()
    encoder.apply(initialize_weights)
    encoder.to(device=DEVICE)
    print(encoder)

    decoder = Decoder()
    decoder.train()
    decoder.apply(initialize_weights)
    decoder.to(device=DEVICE)
    print(decoder)

    critic = Critic()
    critic.train()
    critic.apply(initialize_weights)
    critic.to(device=DEVICE)

    optim_critic = optim.RMSprop(critic.parameters(), lr=0.00005)
    optim_encoder = optim.RMSprop(encoder.parameters(), lr=0.00005)
    optim_decoder = optim.AdamW(decoder.parameters(), lr=0.005)
    loss_function = nn.MSELoss(reduction="sum")
    writer = SummaryWriter("runs/GAN/test")

    loss_encoder_arr = []
    loss_decoder_arr = []
    loss_critic_arr = []
    for e in range(epoch):
        print(f"epoch: {e+1}")
        for iter, (target, label) in enumerate(tqdm(Dataloader)):
            target = target.to(device=DEVICE)
            label = label.to(device=DEVICE)
            torch.div(label, 10)

            # train critic
            for _ in range(7):
                random = torch.rand((BATCH, 9), device=DEVICE)
                critic_rand = critic(random)

                hidden_critic = encoder(target).reshape(BATCH, -1)
                hidden_critic = torch.cat(
                    (hidden_critic, label.reshape(BATCH, 1)), 1)
                critic_hidden = critic(hidden_critic).reshape(-1)

                # wasserstein loss
                loss_critic = -(torch.mean(critic_rand) -
                                torch.mean(critic_hidden))

                loss_encoder_arr.append(loss_critic.item())
                for param in critic.parameters():
                    param.grad = None
                loss_critic.backward(retain_graph=True)
                optim_critic.step()

                # weight clipping
                for p in critic.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # train encoder
            encoded = encoder(target).reshape(BATCH, -1)
            critic_output = critic(torch.cat(
                (encoded, label.reshape(BATCH, 1)), 1))

            loss_encoder = -torch.mean(critic_output)
            loss_decoder_arr.append(loss_encoder.item())

            for param in encoder.parameters():
                param.grad = None
            loss_encoder.backward()
            optim_encoder.step()

            # train decoder
            hidden_decoder = encoder(target).reshape(BATCH, -1)
            hidden_decoder = torch.cat(
                (hidden_decoder, label.reshape(BATCH, 1)), 1).reshape(BATCH, 9, 1, 1)
            decoded = decoder(hidden_decoder)
            loss_decoder = loss_function(decoded, target)
            loss_critic_arr.append(loss_decoder.item())

            for param in decoder.parameters():
                param.grad = None
            loss_decoder.backward()
            optim_decoder.step()

            global_step = e*len(Dataloader)+iter
            logScalarToTensorboard(
                lc=loss_critic_arr, ld=loss_decoder_arr, le=loss_encoder_arr, step=global_step, writer=writer)

            if global_step % 1000 == 0 or global_step == 1:
                logImageToTensorboard(decoder=decoder, encoder=encoder, label=label,
                                      step=global_step, target=target, writer=writer)
                loss_encoder_arr = []
                loss_decoder_arr = []
                loss_critic_arr = []

        print("----------------\n\rsaving...")
        torch.save(encoder, PATH + "Enet.pth")
        torch.save(decoder, PATH + "Dnet.pth")
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


def logImageToTensorboard(writer, step, encoder, decoder, target, label):
    with torch.no_grad():
        test = decoder(torch.cat(
            (encoder(target).reshape(BATCH, -1), label.reshape(BATCH, 1)), 1).reshape(BATCH, 9, 1, 1))

        test_grid = torchvision.utils.make_grid(
            test[:BATCH], normalize=True
        )

        writer.add_image("TestImage", test_grid, global_step=step)

        real_grid = torchvision.utils.make_grid(
            target[:BATCH], normalize=True
        )

        writer.add_image("RealImage", real_grid, global_step=step)


def start():
    train(50)


if __name__ == "__main__":
    start()

    if torch.cuda.is_available():
        print("---\r\nGPU MODE\r\n---")
    else:
        print("---\r\nCPU MODE\r\n---")
