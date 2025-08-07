# 在古腾堡计划数据集上预训练GPT

此目录中的代码包含在古腾堡计划提供的免费书籍上训练小型GPT模型的代码。

正如古腾堡计划网站所述，"古腾堡计划电子书的绝大多数在美国属于公共领域。"

请阅读[古腾堡计划权限、许可和其他常见请求](https://www.gutenberg.org/policy/permission.html)页面，了解更多关于使用古腾堡计划提供资源的信息。

&nbsp;
## 如何使用此代码

&nbsp;

### 1) 下载数据集

在本节中，我们使用来自[`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) GitHub仓库的代码从古腾堡计划下载书籍。

截至撰写本文时，这将需要大约50 GB的磁盘空间，并需要大约10-15小时，但可能会更多，这取决于古腾堡计划自那时以来增长了多少。

&nbsp;
#### Linux和macOS用户的下载说明


Linux和macOS用户可以按照以下步骤下载数据集（如果您是Windows用户，请参见下面的说明）：

1. 将`03_bonus_pretraining_on_gutenberg`文件夹设置为工作目录，以在此文件夹中本地克隆`gutenberg`仓库（这是运行提供的脚本`prepare_dataset.py`和`pretraining_simple.py`所必需的）。例如，当在`LLMs-from-scratch`仓库文件夹中时，通过以下方式导航到*03_bonus_pretraining_on_gutenberg*文件夹：
```bash
cd ch05/03_bonus_pretraining_on_gutenberg
```

2. 在那里克隆`gutenberg`仓库：
```bash
git clone https://github.com/pgcorpus/gutenberg.git
```

3. 导航到本地克隆的`gutenberg`仓库文件夹：
```bash
cd gutenberg
```

4. 从`gutenberg`仓库文件夹安装*requirements.txt*中定义的所需包：
```bash
pip install -r requirements.txt
```

5. 下载数据：
```bash
python get_data.py
```

6. 返回到`03_bonus_pretraining_on_gutenberg`文件夹
```bash
cd ..
```

&nbsp;
#### Windows用户的特殊说明

[`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg)代码与Linux和macOS兼容。但是，Windows用户需要进行小的调整，例如在`subprocess`调用中添加`shell=True`并替换`rsync`。

另外，在Windows上运行此代码的更简单方法是使用"Windows子系统Linux"（WSL）功能，该功能允许用户在Windows中使用Ubuntu运行Linux环境。有关更多信息，请阅读[Microsoft的官方安装说明](https://learn.microsoft.com/en-us/windows/wsl/install)和[教程](https://learn.microsoft.com/en-us/training/modules/wsl-introduction/)。

使用WSL时，请确保您已安装Python 3（通过`python3 --version`检查，或例如使用`sudo apt-get install -y python3.10`安装Python 3.10）并在那里安装以下包：

```bash
sudo apt-get update && \
sudo apt-get upgrade -y && \
sudo apt-get install -y python3-pip && \
sudo apt-get install -y python-is-python3 && \
sudo apt-get install -y rsync
```

> **注意：**
> 关于如何设置Python和安装包的说明可以在[可选Python设置首选项](../../setup/01_optional-python-setup-preferences/README.md)和[安装Python库](../../setup/02_installing-python-libraries/README.md)中找到。
>
> 可选地，此仓库提供了运行Ubuntu的Docker镜像。关于如何使用提供的Docker镜像运行容器的说明可以在[可选Docker环境](../../setup/03_optional-docker-environment/README.md)中找到。

&nbsp;
### 2) 准备数据集

接下来，运行`prepare_dataset.py`脚本，该脚本将（截至撰写本文时的60,173个）文本文件连接成较少的较大文件，以便可以更有效地传输和访问：

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

```
...
Skipping gutenberg/data/raw/PG29836_raw.txt as it does not contain primarily English text.                                     Skipping gutenberg/data/raw/PG16527_raw.txt as it does not contain primarily English text.                                     100%|██████████████████████████████████████████████████████████| 57250/57250 [25:04<00:00, 38.05it/s]
42 file(s) saved in /Users/sebastian/Developer/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg/gutenberg_preprocessed
```


> **提示：**
> 请注意，生成的文件以纯文本格式存储，为了简单起见没有预分词。但是，如果您计划更频繁地使用数据集或训练多个epoch，您可能希望更新代码以预分词形式存储数据集以节省计算时间。有关更多信息，请参见本页底部的*设计决策和改进*。

> **提示：**
> 您可以选择较小的文件大小，例如50 MB。这将产生更多文件，但对于在少量文件上进行快速预训练运行以进行测试可能很有用。


&nbsp;
### 3) 运行预训练脚本

您可以按如下方式运行预训练脚本。请注意，为了说明目的，显示了带有默认值的附加命令行参数：

```bash
python pretraining_simple.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

输出将按以下方式格式化：

> Total files: 3
> Tokenizing file 1 of 3: data_small/combined_1.txt
> Training ...
> Ep 1 (Step 0): Train loss 9.694, Val loss 9.724
> Ep 1 (Step 100): Train loss 6.672, Val loss 6.683
> Ep 1 (Step 200): Train loss 6.543, Val loss 6.434
> Ep 1 (Step 300): Train loss 5.772, Val loss 6.313
> Ep 1 (Step 400): Train loss 5.547, Val loss 6.249
> Ep 1 (Step 500): Train loss 6.182, Val loss 6.155
> Ep 1 (Step 600): Train loss 5.742, Val loss 6.122
> Ep 1 (Step 700): Train loss 6.309, Val loss 5.984
> Ep 1 (Step 800): Train loss 5.435, Val loss 5.975
> Ep 1 (Step 900): Train loss 5.582, Val loss 5.935
> ...
> Ep 1 (Step 31900): Train loss 3.664, Val loss 3.946
> Ep 1 (Step 32000): Train loss 3.493, Val loss 3.939
> Ep 1 (Step 32100): Train loss 3.940, Val loss 3.961
> Saved model_checkpoints/model_pg_32188.pth
> Book processed 3h 46m 55s
> Total time elapsed 3h 46m 55s
> ETA for remaining books: 7h 33m 50s
> Tokenizing file 2 of 3: data_small/combined_2.txt
> Training ...
> Ep 1 (Step 32200): Train loss 2.982, Val loss 4.094
> Ep 1 (Step 32300): Train loss 3.920, Val loss 4.097
> ...


&nbsp;
> **提示：**
> 在实践中，如果您使用macOS或Linux，我建议使用`tee`命令将日志输出保存到`log.txt`文件中，同时在终端上打印它们：

```bash
python -u pretraining_simple.py | tee log.txt
```

&nbsp;
> **警告：**
> 请注意，在`gutenberg_preprocessed`文件夹中的一个约500 Mb文本文件上进行训练在V100 GPU上大约需要4小时。
> 该文件夹包含47个文件，大约需要200小时（超过1周）才能完成。您可能希望在较少数量的文件上运行它。


&nbsp;
## 设计决策和改进

请注意，此代码专注于保持简单和最小化以用于教育目的。代码可以通过以下方式改进以提高建模性能和训练效率：

1. 修改`prepare_dataset.py`脚本以从每个书籍文件中删除古腾堡样板文本。
2. 更新数据准备和加载实用程序以预分词数据集并以分词形式保存，这样在调用预训练脚本时就不必每次重新分词。
3. 通过添加[附录D：为训练循环添加花里胡哨的功能](../../appendix-D/01_main-chapter-code/appendix-D.ipynb)中介绍的功能来更新`train_model_simple`脚本，即余弦衰减、线性预热和梯度裁剪。
4. 更新预训练脚本以保存优化器状态（参见第5章中的*5.4 在PyTorch中加载和保存权重*部分；[ch05.ipynb](../../ch05/01_main-chapter-code/ch05.ipynb)）并添加加载现有模型和优化器检查点的选项，如果训练运行被中断则继续训练。
5. 添加更高级的记录器（例如，Weights and Biases）以实时查看损失和验证曲线
6. 添加分布式数据并行（DDP）并在多个GPU上训练模型（参见附录A中的*A.9.3 使用多个GPU训练*部分；[DDP-script.py](../../appendix-A/01_main-chapter-code/DDP-script.py)）。
7. 将`previous_chapter.py`脚本中的从零开始的`MultiheadAttention`类替换为在[高效多头注意力实现](../../ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)奖励部分中实现的高效`MHAPyTorchScaledDotProduct`类，该类通过PyTorch的`nn.functional.scaled_dot_product_attention`函数使用Flash Attention。
8. 通过[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)（`model = torch.compile`）或[thunder](https://github.com/Lightning-AI/lightning-thunder)（`model = thunder.jit(model)`）优化模型来加速训练。
9. 实现梯度低秩投影（GaLore）以进一步加速预训练过程。这可以通过将`AdamW`优化器替换为[GaLore Python库](https://github.com/jiaweizzhao/GaLore)中提供的`GaLoreAdamW`来实现。
