# CS420-Course-Project
SJTU, CS420, 2019-2020 Spring, Group 14 
# pix2pix

## train
python train.py --dataroot xxx --name xxxx --model pix2pix --direction AtoB
## test
python test.py --dataroot xxx --name xxxx --model pix2pix --direction AtoB
## data preparation
Create folder /path/to/data with subfolders A and B. A and B should each have their own subfolders train, val, test, etc. In /path/to/data/A/train, put training images in style A. In /path/to/data/B/train, put the corresponding images in style B. Repeat same for other data splits (val, test, etc).
Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., /path/to/data/A/train/1.jpg is considered to correspond to /path/to/data/B/train/1.jpg.
Once the data is formatted this way, call:

python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data

This will combine each pair of images (A,B) into a single image file, ready for training.

## Acknowledgement
The code is learnt from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
