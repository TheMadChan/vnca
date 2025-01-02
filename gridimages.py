import os
from PIL import Image
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB')  # Convert to RGB
            images.append(img)
    return images

def create_image_grid(images, grid_size=(4, 4), image_size=(64, 64)):
    """
    Create a grid of images using Matplotlib.

    Args:
        images (list of PIL.Image): A list of PIL images.
        grid_size (tuple): Number of rows and columns in the grid.
        image_size (tuple): Size of each image in the grid.

    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(image_size[0] * grid_size[1] / 100, image_size[1] * grid_size[0] / 100))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].resize(image_size)
            ax.imshow(img)
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axes

def save_image_grid(fig, filename):
    """
    Save the image grid to a file.

    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure object.
        filename (str): The filename to save the image grid.
    """
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)

# Example usage
if __name__ == "__main__":
    folder = r'D:\Academic\ETH\SciML\Chan_SciML\vnca\generated_fractures\latest_n10000_b32_tb32_z128_t0.75_lr0.001_beta1.0_augTrue\growth'

    # Load images from the folder
    images = load_images_from_folder(folder)

    # Create the image grid
    fig, axes = create_image_grid(images, grid_size=(11, 5), image_size=(64, 64))

    # Save the image grid to a file
    save_image_grid(fig, "image_grid_64.png")