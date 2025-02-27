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


# Reference:

@article{wu2024deep,
  title={Deep joint source-channel coding for adaptive image transmission over MIMO channels},
  author={Wu, Haotian and Shao, Yulin and Bian, Chenghong and Mikolajczyk, Krystian and G{\"u}nd{\"u}z, Deniz},
  journal={IEEE Transactions on Wireless Communications},
  year={2024},
  publisher={IEEE}
}

@article{wu2025deep,
  title={Deep Joint Source and Channel Coding},
  author={Wu, Haotian and Bian, Chenghong and Shao, Yulin and G{\"u}nd{\"u}z, Deniz},
  journal={Foundations of Semantic Communication Networks},
  pages={61--110},
  year={2025},
  publisher={Wiley Online Library}
}
