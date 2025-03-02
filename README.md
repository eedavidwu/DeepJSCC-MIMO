# Notes for Codes

## Running the Code

Run the following command:

```bash
python train.py
```

## Settings

- Select different **patch sizes** and their **positional encoding settings** for different data resolutions.
- Select different `args.tcn` for different **bandwidth ratios**.

## Block Selection

- For **CIFAR-10** and **CelebA**, this **Seq-to-Seq architecture** performs well with best performance, especially under lower complexity.
- Replacing the **learnable positional encoding** with **conditional positional encoding** benefits **resolution-adaptive design**.

## Some Tips

- For **large-resolution settings**, using an **alternative SWIN block** can speed up training.  
  _(Will update for convenient usage when available.)_
- Training time can be optimized:  
  - Current training is slow due to **SVD decomposition**.  
  - Consider accelerating with **GPU support**.
- If the GPU memoary is not allowed, another simple way is to train across 128x128 patches, and reconstruct image in patch-wise and then compute PSNR. (Still can match the similar SOTA performance.)


> **Note**: The code is currently uncleaned. I will clean it and make it more user-friendly when I have time.  

## Reference

```bibtex
@article{wu2024deep,
  title={Deep joint source-channel coding for adaptive image transmission over MIMO channels},
  author={Wu, Haotian and Shao, Yulin and Bian, Chenghong and Mikolajczyk, Krystian and G{"u}nd{"u}z, Deniz},
  journal={IEEE Transactions on Wireless Communications},
  year={2024},
  publisher={IEEE}
}

@article{wu2025deep,
  title={Deep Joint Source and Channel Coding},
  author={Wu, Haotian and Bian, Chenghong and Shao, Yulin and G{"u}nd{"u}z, Deniz},
  journal={Foundations of Semantic Communication Networks},
  pages={61--110},
  year={2025},
  publisher={Wiley Online Library}
}
