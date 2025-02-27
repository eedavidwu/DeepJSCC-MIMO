# Notes for codes:

Run python train.py

# Setting:

Select different patch sizes and their positional encoding settings for different data resolutions.

Select different args.tcn for different bandwidth ratios.

# Block selection:

For Cifar10 and CelebA, this Seq-to-Seq architecture works very well, with nice performance under lower complexity. 

Replacing the learnable positional encoding with conditional positional encoding can benefit a lot for resolution-adaptive design. 

# Some tips:

For a very large-resolution setting, an alternative SWIN block can make the training faster. I will update it when I have time for convenient usage.

Another thing that can be optimized is training time. It is a bit slow due to SVD-decomposition, which can be accelerated with GPU.

(This code is uncleaned, I will also clean the code and make it easily used when free.)
