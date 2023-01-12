# CycleGAN and pix2pix in PyTorch

## 1、Notebook教程

PyTorch Colab notebook教程: [CycleGAN](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb) 和 [pix2pix](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb)

## 2、环境要求

- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## 3、开始实用

### 3.1、安装

- 克隆本仓库:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies.
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

### 3.2、CycleGAN训练和测试
- 下载一个CycleGAN数据集:
```bash
bash ./datasets/download_cyclegan_dataset.sh maps
```
- 查看训练结果和损失图, 玉兴 `python -m visdom.server` 并且点击打开 URL http://localhost:8097.
- 将训练进度和测试图像记录到W&B仪表板上, 设置 `--use_wandb` 带有训练和测试脚本的标志
- 训练模型:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
要看到更多的中间结果, 查看 `./checkpoints/maps_cyclegan/web/index.html`.
- 测试模型:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
- 测试结果将被保存在这里的一个html文件中: `./results/maps_cyclegan/latest_test/index.html`.

### 3.3、pix2pix训练和测试
- 下载一个pix2pix数据集:
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- 查看训练结果和损失图, 运行 `python -m visdom.server` 并且点击打开URL http://localhost:8097.
- 将训练进度和测试图像记录到W&B仪表板上, 设置 `--use_wandb` 带有训练和测试脚本的标志
- 训练模型:
```bash
#!./scripts/train_pix2pix.sh
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
要看到更多的中间结果, 查看  `./checkpoints/facades_pix2pix/web/index.html`.

- 测试模型 (`bash ./scripts/test_pix2pix.sh`):
```bash
#!./scripts/test_pix2pix.sh
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
- 测试结果将被保存在这里的一个html文件中: `./results/facades_pix2pix/test_latest/index.html`. 你可以在`scripts`目录下找到更多的脚本。
- 为了训练和测试基于pix2pix的着色模型，请添加`--model colorization`和`--dataset_mode colorization`。 查看我们的训练 [tips](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#notes-on-colorization) 关于更多细节.

### 3.4、使用预训练模型 (CycleGAN)
- 你可以用下面的脚本下载一个预训练过的模型（如horse2zebra）：
```bash
bash ./scripts/download_cyclegan_model.sh horse2zebra
```
- The pretrained model is saved at `./checkpoints/{name}_pretrained/latest_net_G.pth`. Check [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_cyclegan_model.sh#L3) for all the available CycleGAN models.
- 为了测试该模型，你还需要下载horse2zebra数据集。
```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

- 然后用以下方法生成结果
```bash
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
```
- The option `--model test` is used for generating results of CycleGAN only for one side. This option will automatically set `--dataset_mode single`, which only loads the images from one set. On the contrary, using `--model cycle_gan` requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at `./results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory.

- 对于pix2pix和你自己的模型，你需要明确指定`--netG`、`--norm`、`--no_dropout`以匹配训练过的模型的生成器结构。更多细节请看这个[FAQ](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md#runtimeerror-errors-in-loading-state_dict-812-671461-296)。

### 3.5、使用预训练模型 (pix2pix)
下载预训练模型使用 `./scripts/download_pix2pix_model.sh`.

- Check [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_pix2pix_model.sh#L3) for all the available pix2pix models. For example, if you would like to download label2photo model on the Facades dataset,
```bash
bash ./scripts/download_pix2pix_model.sh facades_label2photo
```
- 下载pix2pix facades数据集
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- 然后用以下方法生成结果
```bash
python test.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained
```

- 请注意，我们指定了`--direction BtoA`，因为Facades数据集的A到B方向是照片到标签。

- 如果你想把预先训练好的模型应用于输入图像的集合（而不是图像对），请使用`--model test`选项。参见`./scripts/test_single.sh`，了解如何将模型应用于Facade标签图（存储在`facades/testB`目录下）。

- 请参阅目前可用的模型列表 `./scripts/download_pix2pix_model.sh`

## 3.6、[数据集](docs/datasets.md)
下载pix2pix/CycleGAN数据集并创建你自己的数据集。

## 3.7、[训练和测试Tips](docs/tips.md)
训练和测试你的模型的最佳做法。

## 3.8、[常见问题](docs/qa.md)
在你发布新问题之前，请先看看上述问答和现有的GitHub问题。

## 3.9、自定义模型和数据集
如果你打算为你的新应用程序实现自定义模型和数据集，我们提供了一个数据集[模板](data/template_dataset.py)和一个模型[模板](models/template_model.py)作为起点。

## 3.10、[Code structure](docs/overview.md)
为了帮助用户更好地理解和使用我们的代码，我们简要地概述了每个包和每个模块的功能和实现。

## 3.11、参考

- [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)


