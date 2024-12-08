import os
from typing import Sequence, Tuple
import argparse

import numpy as np
import torch as t
import torch.utils.data
from PIL import Image
from shapeguard import ShapeGuard
from torch import nn, optim
from torch.distributions import Normal, Distribution, kl_divergence, Bernoulli
from torch.utils.data import DataLoader, ConcatDataset
from distutils_fallback import LooseVersion
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard._utils import make_grid
import torchvision.utils as vutils
import tqdm
import torchvision.transforms as transforms


from data.mnist import StaticMNIST
from data.fractures import FractureImages
from data.augmentation import augment_dataset, rotation_transforms, flip_transforms
from iterable_dataset_wrapper import IterableWrapper
from modules.model import Model
from modules.nca import MitosisNCA
from modules.residual import Residual
from train import train
from util import get_writers


# torch.autograd.set_detect_anomaly(True)

class VAENCA(Model, nn.Module):
    def __init__(self, ds = 'fractures', n_updates = 100, batch_size=32, test_batch_size = 32, z_size=128, bin_threshold=0.75, learning_rate=1e-4, beta=1.0, augment=False):  
        super(Model, self).__init__()
        self.h = self.w = 32
        self.z_size = z_size
        self.train_loss_fn = self.elbo_loss_function
        self.train_samples = 1
        self.test_loss_fn = self.iwae_loss_fn
        self.test_samples = 1
        self.nca_hid = z_size
        self.encoder_hid = 32
        self.beta = beta
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.bin_threshold = bin_threshold
        self.augment = augment
        self.learning_rate = learning_rate
        self.n_updates = n_updates
        self.bpd_dimensions = 1 * 32 * 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        # self.device = "cpu" if ds == "mnist" else "cuda" if torch.cuda.is_available() else "cpu"


        filter_size = (5, 5)
        pad = tuple(s // 2 for s in filter_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.encoder_hid * 2 ** 0, filter_size, padding=pad), nn.ELU(),  # (bs, 32, h, w)
            nn.Conv2d(self.encoder_hid * 2 ** 0, self.encoder_hid * 2 ** 1, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 64, h//2, w//2)
            nn.Conv2d(self.encoder_hid * 2 ** 1, self.encoder_hid * 2 ** 2, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 128, h//4, w//4)
            nn.Conv2d(self.encoder_hid * 2 ** 2, self.encoder_hid * 2 ** 3, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 256, h//8, w//8)
            nn.Conv2d(self.encoder_hid * 2 ** 3, self.encoder_hid * 2 ** 4, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 512, h//16, w//16),
            nn.Flatten(),  # (bs, 512*h//16*w//16)
            nn.Linear(self.encoder_hid * (2 ** 4) * self.h // 16 * self.w // 16, 2 * self.z_size),
        )

        update_net = t.nn.Sequential(
            t.nn.Conv2d(self.z_size, self.nca_hid, 3, padding=1),
            Residual(
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
                t.nn.ELU(),
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
            ),
            Residual(
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
                t.nn.ELU(),
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
            ),
            Residual(
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
                t.nn.ELU(),
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
            ),
            Residual(
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
                t.nn.ELU(),
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
            ),
            t.nn.Conv2d(self.nca_hid, self.z_size, 1)
        )
        update_net[-1].weight.data.fill_(0.0)
        update_net[-1].bias.data.fill_(0.0)

        self.nca = MitosisNCA(self.h, self.w, self.z_size, update_net, int(np.log2(self.h)) - 1, 8, 1.0, device=self.device)
        self.p_z = Normal(t.zeros(self.z_size, device=self.device), t.ones(self.z_size, device=self.device))

        if ds == 'mnist':
            # self.device = "cpu"
            data_dir = os.environ.get('DATA_DIR') or "."
            train_data, val_data = StaticMNIST(data_dir, 'train'), StaticMNIST(data_dir, 'val'),
            self.test_set = StaticMNIST(data_dir, 'test')
        else:
            # Define transformations
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),  # Resize to 32x32 if needed
                transforms.ToTensor(),        # Convert to tensor
                transforms.Lambda(lambda x: (x > bin_threshold).float())  # Binarize the image
                ])
            
            data_dir = "../glass_fractures"
            train_data, val_data = FractureImages(data_dir, 'train', transform=self.transform), FractureImages(data_dir, 'validation', transform=self.transform),
            self.test_set = FractureImages(data_dir, 'test', transform=self.transform)
            
            if augment:
                # Augment the dataset
                train_augmented = augment_dataset(train_data, rotation_transforms, flip_transforms)
                val_augmented = augment_dataset(val_data, rotation_transforms, flip_transforms)

                train_data = ConcatDataset((train_data, train_augmented))
                val_data = ConcatDataset((val_data, val_augmented))

        train_data = ConcatDataset((train_data, val_data))

        self.train_loader = iter(DataLoader(IterableWrapper(train_data), batch_size=batch_size, pin_memory=True))
        self.test_loader = iter(DataLoader(IterableWrapper(self.test_set), batch_size=test_batch_size, shuffle=False, pin_memory=True))
        self.train_writer, self.test_writer = get_writers("hierarchical-nca", ds, n_updates, batch_size, test_batch_size, z_size, bin_threshold, learning_rate, self.beta)

        print(self)
        for n, p in self.named_parameters():
            print(n, p.shape)

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.batch_idx = 0

    def train_batch(self):
        self.train(True)

        self.optimizer.zero_grad()
        x, y = next(self.train_loader)
        loss, z, p_x_given_z, recon_loss, kl_loss = self.forward(x, self.train_samples, self.train_loss_fn)
        loss.backward()

        t.nn.utils.clip_grad_norm_(self.parameters(), 10.0)

        self.optimizer.step()

        if self.batch_idx % 100 == 0:
            self.report(self.train_writer, p_x_given_z, loss, recon_loss, kl_loss)

        self.batch_idx += 1
        return loss.item()

    def save(self, fn):
        t.save({
            'batch_idx': self.batch_idx,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, fn)

    def load(self, fn):
        checkpoint = t.load(fn, map_location=t.device(self.device))
        self.batch_idx = checkpoint["batch_idx"]
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def eval_batch(self):
        self.train(False)
        with t.no_grad():
            # x, y = next(self.test_loader)
            x = self.test_set.samples
            loss, z, p_x_given_z, recon_loss, kl_loss = self.forward(x, self.test_samples, self.test_loss_fn)
            self.report(self.test_writer, p_x_given_z, loss, recon_loss, kl_loss)
        return loss.item()

    def test(self, n_iw_samples):
        self.train(False)
        with t.no_grad():
            total_loss = 0.0
            for x in tqdm.tqdm(self.test_set.samples):
                x = x.unsqueeze(0)
                loss, z, p_x_given_z, recon_loss, kl_loss = self.forward(x, n_iw_samples, self.test_loss_fn)
                total_loss += loss

        print(f"total iwae-loss is -{total_loss / len(self.test_set)}nats")

    def _plot_samples(self):
        ShapeGuard.reset()
        with torch.no_grad():
            samples = self.p_z.sample((64, 1)).to(self.device)
            decode, states = self.decode(samples)
            samples = self.to_rgb(states[-1])
            # rgb=0.3, alpha=0 --> samples = 1-0+0.3*0 = 1 = white
            # rgb = 0.3, alpha=1 --> samples = 1-1+0.3*1 = 0.3
            # rgb = 0.3, alpha=0.5, samples = 1-0.5+0.3*0.5 = 0.5+0.15 = 0.65

            growth = []
            for state in states:
                rgb = self.to_rgb(state[0:1])
                h = state.shape[3]
                pad = (self.h - h) // 2
                rgb = t.nn.functional.pad(rgb, [pad] * 4, mode="constant", value=0)
                growth.append(rgb)
            growth = t.cat(growth, dim=0).cpu().detach().numpy()  # (n_states, 3, h, w)

            # x, y = next(self.test_loader)
            x = self.test_set.samples
            ground_truth = x[:64]
            _, _, p_x_given_z, _, _ = self.forward(x[:64], 1, self.iwae_loss_fn)
            recons = self.to_rgb(p_x_given_z.logits.reshape(-1, 1, self.h, self.w))

        return samples, recons, growth, ground_truth

    def to_rgb(self, samples):
        return Bernoulli(logits=samples[:, :1, :, :]).sample()

    def plot_growth_samples(self):
        ShapeGuard.reset()
        with torch.no_grad():
            samples = self.p_z.sample((64, 1)).to(self.device)
            _, states = self.decode(samples)
            for i, state in enumerate(states):
                samples = t.clip(state[:, :4, :, :], 0, 1).cpu().detach().numpy()
                samples = self.to_rgb(samples)
                samples = (samples * 255).astype(np.uint8)
                grid = make_grid(samples).transpose(1, 2, 0)  # (HWC)
                im = Image.fromarray(grid)
                im.save("samples-%03d.png" % i)

    def report(self, writer: SummaryWriter, p_x_given_z, loss, recon_loss, kl_loss):
        writer.add_scalar('loss', loss.item(), self.batch_idx)
        writer.add_scalar('bpd', loss.item() / (np.log(2) * self.bpd_dimensions), self.batch_idx)
        writer.add_scalar('entropy', p_x_given_z.entropy().mean().item(), self.batch_idx)
        if recon_loss:
            writer.add_scalar('recon_loss', recon_loss.item(), self.batch_idx)
        if kl_loss:
            writer.add_scalar('kl_loss', kl_loss.item(), self.batch_idx)

        samples, recons, growth, ground_truth = self._plot_samples()
        
        # Add padding to the samples tensor
        padding = 1  # Amount of padding to add to each side
        samples = torch.nn.functional.pad(samples, (padding, padding, padding, padding), mode='constant', value=0)
        recons = torch.nn.functional.pad(recons, (padding, padding, padding, padding), mode='constant', value=0)
        # growth = torch.nn.functional.pad(growth, (padding, padding, padding, padding), mode='constant', value=0)
        ground_truth = torch.nn.functional.pad(ground_truth, (padding, padding, padding, padding), mode='constant', value=0)

        # writer.add_images("grid", grid, self.batch_idx)
        writer.add_images("samples", samples, self.batch_idx)
        writer.add_images("recons", recons, self.batch_idx)
        writer.add_images("ground_truth", ground_truth, self.batch_idx)
        writer.add_images("growth", growth, self.batch_idx)

    def encode(self, x) -> Distribution:  # q(z|x)
        x.sg("B4hw")
        q = self.encoder(x).sg("BZ")
        loc = q[:, :self.z_size].sg("Bz")
        logsigma = q[:, self.z_size:].sg("Bz")
        return Normal(loc=loc, scale=t.exp(logsigma))

    def decode(self, z: t.Tensor) -> Tuple[Distribution, Sequence[t.Tensor]]:  # p(x|z)
        z = z.to(self.device)
        z.sg("Bnz")
        bs, ns, zs = z.shape
        state = z.reshape((-1, self.z_size)).unsqueeze(2).unsqueeze(3).expand(-1, -1, 2, 2).sg("bz22")
        states = self.nca(state)

        state = states[-1]

        # logits = state[:, :1, :, :].sg("b1hw").reshape((bs, ns, -1)).sg("Bnx")
        logits = state[:, :1, :, :].reshape((bs, ns, -1)).sg("Bnx")

        return Bernoulli(logits=logits), states

    def forward(self, x, n_samples, loss_fn):
        ShapeGuard.reset()
        x.sg("B4hw")
        x = x.to(self.device)
        x_flat = x.reshape(x.shape[0], -1).sg("Bx")
        q_z_given_x: Distribution = self.encode(x).sg("Bz")
        z = q_z_given_x.rsample((n_samples,)).permute((1, 0, 2)).sg("Bnz")
        decode, _ = self.decode(z)
        p_x_given_z = decode.sg("Bnx")

        loss, recon_loss, kl_loss = loss_fn(x_flat, p_x_given_z, q_z_given_x, z)
        return loss, z, p_x_given_z, recon_loss, kl_loss

    def iwae_loss_fn(self, x: t.Tensor, p_x_given_z: Distribution, q_z_given_x: Distribution, z: t.Tensor):
        """
          log(p(x)) >= logsumexp_{i=1}^N[ log(p(x|z_i)) + log(p(z_i)) - log(q(z_i|x))] - log(N)
        """
        x.sg("Bx")
        p_x_given_z.sg("Bnx")
        q_z_given_x.sg("Bz")
        z.sg("Bnz")

        logpx_given_z = p_x_given_z.log_prob(x.unsqueeze(1).expand_as(p_x_given_z.mean)).sum(dim=2).sg("Bn")
        logpz = self.p_z.log_prob(z).sum(dim=2).sg("Bn")
        logqz_given_x = q_z_given_x.log_prob(z.permute((1, 0, 2))).sum(dim=2).permute((1, 0)).sg("Bn")
        logpx = (t.logsumexp(logpx_given_z + logpz - logqz_given_x, dim=1) - t.log(t.scalar_tensor(z.shape[1]))).sg("B")
        return -logpx.mean(), None, None  # (1,)

    def elbo_loss_function(self, x: t.Tensor, p_x_given_z: Distribution, q_z_given_x: Distribution, z: t.Tensor):
        """
          log p(x) >= E_q(z|x) [ log p(x|z) p(z) / q(z|x) ]
          Reconstruction + KL divergence losses summed over all elements and batch
        """
        x.sg("Bx")
        p_x_given_z.sg("Bnx")
        q_z_given_x.sg("Bz")
        z.sg("Bnz")

        logpx_given_z = p_x_given_z.log_prob(x.unsqueeze(1).expand_as(p_x_given_z.mean)).sum(dim=2).mean(dim=1).sg("B")
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1).sg("B")

        reconstruction_loss = -logpx_given_z.mean()
        kl_loss = kld.mean()
        loss = reconstruction_loss + self.beta*kl_loss
        return loss, reconstruction_loss, kl_loss  # (1,)
    
    def generate_fracture(self, image: t.Tensor, target_image_size: int) -> t.Tensor:
        self.eval()
        with torch.no_grad():

            transform = self.transform
            image = transform(image)  # Convert to tensor and add batch dimension
            image = image.to(self.device)

            z = self.encode(image.unsqueeze(0)).rsample((1,)).permute((1, 0, 2)).sg("Bnz")

            self.nca.n_duplications = int(np.log2(target_image_size)) - 1
            decode, _ = self.decode(z)

            output_image = decode.sample().reshape(1, target_image_size, target_image_size)  # Sample from the Bernoulli distribution

        return output_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VAE-NCA Training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--z_size', type=int, default=128, help='Size of latent space')
    parser.add_argument('--bin_threshold', type=float, default=0.75, help='Threshold for binarizing the image')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--n_updates', type=int, default=100, help='Number of updates to train the model')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--dataset', type=str, default='fractures', help='Glass fractures or MNIST dataset')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for the KL divergence term')
    parser.add_argument('--augment', type=bool, default=False, help='Augment dataset or not')
    args = parser.parse_args()

    model = VAENCA(ds=args.dataset, 
                   n_updates=args.n_updates, 
                   batch_size=args.batch_size, 
                   test_batch_size = args.test_batch_size, 
                   z_size=args.z_size, 
                   bin_threshold=args.bin_threshold, 
                   learning_rate=args.learning_rate,
                   beta=args.beta,
                   augment=args.augment)
    
    model.eval_batch()
    
    train(model, n_updates=args.n_updates, eval_interval=100)
    
    model.test(128)

