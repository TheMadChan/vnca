import torchvision.transforms as transforms

# Define the augmentation transformations
rotation_transforms = [
    transforms.RandomRotation((90, 90)),
    transforms.RandomRotation((180, 180)),
    transforms.RandomRotation((270, 270)),
]

flip_transforms = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
]

def augment_dataset(dataset, rotation_transforms, flip_transforms):
    augmented_images = []
    for img, label in dataset:
        # Apply flips to the original image
        for flip in flip_transforms:
            flipped_img = flip(img)
            augmented_images.append((flipped_img, label))
        
        # Apply rotations to the original image
        for rotation in rotation_transforms:
            rotated_img = rotation(img)
            augmented_images.append((rotated_img, label))
            
            # Apply flips to each rotated image
            for flip in flip_transforms:
                flipped_rotated_img = flip(rotated_img)
                augmented_images.append((flipped_rotated_img, label))
    return augmented_images

