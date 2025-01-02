import argparse
import torch as t
import torchvision.utils as vutils
import torchvision.transforms as transforms
import os
from PIL import Image
from vae_nca import VAENCA


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate fractures from VAE-NCA')
    parser.add_argument('--image_dir', type=str, default="../glass_fractures/test", help='Path to the input image')
    parser.add_argument('--checkpoint', type=str, default='latest', help='Checkpoint name to load the model from')
    parser.add_argument('--target_image_size', type=int, default=32, help='Target size for the output image')
    parser.add_argument('--index', type=int, default=0, help='Index of the image to generate fractures from')
    
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


    # Initialize the model
    model = VAENCA(ds=args.dataset, 
                   n_updates=args.n_updates, 
                   batch_size=args.batch_size, 
                   test_batch_size = args.test_batch_size, 
                   z_size=args.z_size, 
                   bin_threshold=args.bin_threshold, 
                   learning_rate=args.learning_rate,
                   beta=args.beta,
                   augment=args.augment)
    
    checkpoint_name = f"{args.checkpoint}_n{model.n_updates}_b{model.batch_size}_tb{model.test_batch_size}_z{model.z_size}_t{model.bin_threshold}_lr{model.learning_rate}_beta{model.beta}_aug{model.augment}.pt"

    # Load the saved model
    model.load(os.path.join("models", checkpoint_name))

    # Create a directory to save the generated images
    os.makedirs("generated_fractures", exist_ok=True)
    out_images_dir = os.path.join("generated_fractures", checkpoint_name.split('.p')[0])
    os.makedirs(out_images_dir, exist_ok=True)
    growth_images_dir = os.path.join(out_images_dir, "growth")
    os.makedirs(growth_images_dir, exist_ok=True)
    
    # Generate and save samples
    model.generate_and_save_fractures(image_dir=args.image_dir, index=args.index, save_dir=out_images_dir, target_image_size=args.target_image_size, model_name=args.checkpoint)
    # model._plot_samples()
    model.visualize_latent_space(model.latent_loader)
    
    # model.plot_growth_sample_avg(image_dir=args.image_dir, index=args.index, save_dir=out_images_dir, model_name=args.checkpoint)