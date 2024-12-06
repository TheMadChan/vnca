import argparse
import torch
import torchvision.utils as vutils
import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from vae_nca import VAENCA

def generate_and_save_samples(model, n_samples, save_dir="generated_samples"):
    os.makedirs(save_dir, exist_ok=True)
    samples = model.generate_samples(n_samples)
    for i, sample in enumerate(samples):
        vutils.save_image(sample, os.path.join(save_dir, f"sample_{i}.png"))
    print(f"Generated samples saved in {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Samples from VAE-NCA')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--checkpoint', type=str, default='best', help='Checkpoint name to load the model from')
    parser.add_argument('--save_dir', type=str, default='generated_samples', help='Directory to save the generated samples')
    args = parser.parse_args()

    # Initialize the model
    model = VAENCA(ds='fractures', n_updates=10000, batch_size=32, test_batch_size=1, z_size=128, bin_threshold=0.75, learning_rate=1e-4)
    
    # Load the saved model
    model.load(args.checkpoint)
    
    # Generate and save samples
    generate_and_save_samples(model, args.n_samples, args.save_dir)