from pathlib import Path
from os.path import join
import argparse

import scipy.ndimage as ndimage
import numpy as np
import skimage.io as io
import glob
from PIL import Image
from scipy.interpolate import interp1d
from skimage.morphology import disk

# python3 preprocessing.py \
#     --images-path ./data/images \
#     --masks-path ./data/masks \
#     --labels-path ./data/labels \
#     --save-path train/_Data/all/train/


def crop_center(img, cropx, cropy):
    y, x = img.shape[0], img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def to_0_1(img):
    interp_fun = interp1d([img.min(), img.max()], [0.0, 1.0])
    return interp_fun(img)


def enhance_image(img, mask, int_format=False, disk_size=5):
    """Enhance an image using the method described in the paper.
    Args:
        img (np.ndarray): Image to enhance
        mask (np.ndarray): ROI mask of the image to enhance

    Returns:
        Enhanced image
    """
    # Read image and its corresponding mask
    if isinstance(img, str) or isinstance(img, Path):
        img = io.imread(img)[..., :3]
    if isinstance(mask, str) or isinstance(mask, Path):
        mask = io.imread(mask)

    if len(img.shape) == 3:
        if img.shape[2] > 3:
            img = img[:, :, :3]
    if len(mask.shape) == 3:
        mask = np.sum(mask[:, :, :3], axis=2)

    img = img / 255
    # mask = np.where(mask > (255//2), 255, 0)
    mask = np.where(mask > 0.5, 1, 0)

    # Copy original image
    img_copy = img.copy()
    # Convert to PIL format
    zoomed_image = Image.fromarray(np.uint8(img_copy*255))
    # Enlarge image
    zoomed_image = zoomed_image.resize(
        (int(img_copy.shape[1]*1.15), int(img_copy.shape[0]*1.15)),
        Image.BICUBIC
    )
    # To numpy array type
    zoomed_image = np.array(zoomed_image)
    # Crop image to original size (zoom result)
    zoomed_image = crop_center(zoomed_image, img_copy.shape[1],
                               img_copy.shape[0])
    # Convert image from 0-255 format to 0.0-1.0 format
    zoomed_image = zoomed_image / 255.0

    # Create circular kernel for mask erosion
    kernel = disk(disk_size)

    # Erode mask
    mask = ndimage.binary_erosion(mask, kernel)
    # Convert boolean array to float array
    mask = mask * 1.0  # type: ignore

    img_copy[mask < 1.0] = 0.0

    # Create RGB mask (same mask for all channels)
    mask = np.stack((mask, mask, mask), axis=2)

    composed_image = mask.copy()
    composed_image[mask == 1.0] = img_copy[mask == 1.0]
    composed_image[mask < 1.0] = zoomed_image[mask < 1.0]

    filtered_image = ndimage.gaussian_filter(composed_image, sigma=(10, 10, 0))

    subtracted_image = composed_image - filtered_image
    subtracted_image[mask < 1.] = 0.

    enhanced_image = subtracted_image/np.std(subtracted_image)
    enhanced_image = to_0_1(enhanced_image)
    enhanced_image[mask < 1.] = 0.

    mask = mask[:, :, 0]

    if int_format:
        enhanced_image *= 255
        enhanced_image = enhanced_image.astype(np.uint8)
        mask *= 255
        mask = mask.astype(np.uint8)

    return enhanced_image, mask

def preprocess_label(label, mask):
    """
    Zoom + crop + apply eroded mask to labels to match preprocessed images.
    Args:
        label: HxWx3 uint8 RGB label
        mask: HxW eroded ROI mask
    """
    mask_bool = mask > 0
    zoomed_label = Image.fromarray(label)
    zoomed_label = zoomed_label.resize(
        (int(label.shape[1]*1.15), int(label.shape[0]*1.15)),
        Image.NEAREST  # preserve categorical labels
    )
    zoomed_label = np.array(zoomed_label)
    zoomed_label = crop_center(zoomed_label, label.shape[1], label.shape[0])

    # Apply eroded mask
    for c in range(3):
        zoomed_label[:, :, c][~mask_bool] = 0

    return zoomed_label


def enhance_images_labels(image_names, mask_names, label_names, save_path):

    """Enhance a list of images and save them to disk.
    Args:        
        image_names (list): List of paths to the images to enhance
        mask_names (list): List of paths to the masks corresponding to the images
        label_names (list): List of paths to the labels corresponding to the images
        save_path (str): Path to save the enhanced images and labels

    Returns:
        None
    """
    
    for img_path, mask_path, label_path in zip(image_names, mask_names, label_names):

        print(f"Processing {img_path} with mask {mask_path} and label {label_path}...")
        assert Path(img_path).name == Path(mask_path).name == Path(label_path).name
        enhanced_image, eroded_mask = enhance_image(img_path, mask_path, int_format=True)

        # Preprocess label
        label = io.imread(label_path)
        processed_label = preprocess_label(label, eroded_mask)        
        
        # Save enhanced image and label
        io.imsave(join(save_path, 'enhanced', Path(img_path).name), enhanced_image)
        io.imsave(join(save_path, 'av3', Path(label_path).name), processed_label)
        io.imsave(join(save_path, 'enhanced_masks', Path(mask_path).name), eroded_mask)


def main(images_path, masks_path, labels_path, save_path):
    
    image_names = sorted(glob.glob(join(images_path, '*.png')))
    mask_names = sorted(glob.glob(join(masks_path, '*.png')))
    label_names = sorted(glob.glob(join(labels_path, '*.png')))

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / 'enhanced').mkdir(exist_ok=True)
    (save_path / 'av3').mkdir(exist_ok=True)
    (save_path / 'enhanced_masks').mkdir(exist_ok=True)

    enhance_images_labels(image_names, mask_names, label_names, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess images and labels for RRWNet')
    parser.add_argument('--images-path', type=str, required=True)
    parser.add_argument('--masks-path', type=str, required=True)
    parser.add_argument('--labels-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    args = parser.parse_args()
    main(args.images_path, args.masks_path, args.labels_path, args.save_path)
