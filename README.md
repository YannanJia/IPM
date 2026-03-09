<h1 align="center">IPM</h1>

This repository contains the experimental code for the paper:

[Enhancing Adversarial Transferability via Importance-Aware Pixel-Level Mask]()

Yannan Jia, Lize Gu, Shihui Zheng

![Overview of the IPM Attack Pipeline.](./fig/fig.png)


## Requirements

+ Python >= 3.9.2
+ Numpy >= 1.2.4
+ opencv >= 4.1.0
+ scipy > 1.1.1
+ pandas >= 1.2.4
+ imageio >= 2.3.2
+ pytorch == 2.7.0+cu128
+ torchvision == 0.22.0+cu128

```
pip install  requirements.txt
```

## Qucik Start

### Prepare the data and models

You should download the [data](https://drive.google.com/drive/folders/1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS) offered by [Admix](https://github.com/JHL-HUST/Admix) and place it in Input/data with label file `val_rs.csv` in `Input`.

### Runing baseline

Taking baseline for example, you can run this attack as following:

```
python eval-IPM.py 
```

### Runing IPM

Taking IPM for example, you can run this attack as following:

```
python eval-IPM.py --use_mask
```

### Citation

If you find the idea or code useful for your research, please consider citing our [paper]():
```
@article{jia2026enhancing,
  title={Enhancing Adversarial Transferability via Importance-Aware Pixel-Level Mask},
  author={Yannan Jia, Lize Gu, Shihui Zheng},
  year={2026}
}
```
