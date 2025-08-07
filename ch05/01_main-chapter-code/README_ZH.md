# 第5章：在无标签数据上预训练

### 主要章节代码

- [ch05.ipynb](ch05.ipynb) 包含章节中出现的所有代码
- [previous_chapters.py](previous_chapters.py) 是一个Python模块，包含前面章节中的`MultiHeadAttention`模块和`GPTModel`类，我们在[ch05.ipynb](ch05.ipynb)中导入它们来预训练GPT模型
- [gpt_download.py](gpt_download.py) 包含用于下载预训练GPT模型权重的实用函数
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章的练习解答

### 可选代码

- [gpt_train.py](gpt_train.py) 是一个独立的Python脚本文件，包含我们在[ch05.ipynb](ch05.ipynb)中实现的用于训练GPT模型的代码（您可以将其视为总结本章的代码文件）
- [gpt_generate.py](gpt_generate.py) 是一个独立的Python脚本文件，包含我们在[ch05.ipynb](ch05.ipynb)中实现的用于加载和使用OpenAI预训练模型权重的代码
