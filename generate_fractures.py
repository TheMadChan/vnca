import argparse
import torch as t
import torchvision.utils as vutils
import torchvision.transforms as transforms
import os
from PIL import Image
from vae_nca import VAENCA

def generate_and_save_fractures(model, image_dir="../glass_fractures/test", index=0, save_dir="generated_fractures", target_image_size: int = 32, model_name="best"):
    os.makedirs(save_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_name = image_files[index].split('.')[0]

    image_path = os.path.join(image_dir, image_files[index])

    # Load the input image
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    output_image = model.generate_fracture(image, target_image_size)
    vutils.save_image(output_image, os.path.join(save_dir, f"{model_name}_gen_{image_name}_size{target_image_size}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate fractures from VAE-NCA')
    parser.add_argument('--image_dir', type=str, default="../glass_fractures/test", help='Path to the input image')
    parser.add_argument('--checkpoint', type=str, default='latest', help='Checkpoint name to load the model from')
    parser.add_argument('--save_dir', type=str, default='generated_fractures', help='Directory to save the generated samples')
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
    
    # Generate and save samples
    generate_and_save_fractures(model, image_dir=args.image_dir, index=args.index, save_dir=args.save_dir, target_image_size=args.target_image_size, model_name=checkpoint_name.split('.p')[0])