# 第5章：在无标签数据上进行预训练

&nbsp;
## 主要章节代码

- [01_main-chapter-code](01_main-chapter-code) 包含主要章节代码

&nbsp;
## 额外内容

- [02_alternative_weight_loading](02_alternative_weight_loading) 包含从其他地方加载GPT模型权重的代码，以防模型权重从OpenAI变得不可用
- [03_bonus_pretraining_on_gutenberg](03_bonus_pretraining_on_gutenberg) 包含在古腾堡计划的整个书籍语料库上更长时间预训练LLM的代码
- [04_learning_rate_schedulers](04_learning_rate_schedulers) 包含实现更复杂训练函数的代码，包括学习率调度器和梯度裁剪
- [05_bonus_hparam_tuning](05_bonus_hparam_tuning) 包含可选的超参数调优脚本
- [06_user_interface](06_user_interface) 实现与预训练LLM交互的交互式用户界面
- [07_gpt_to_llama](07_gpt_to_llama) 包含将GPT架构实现转换为Llama 3.2的分步指南，并从Meta AI加载预训练权重
- [08_memory_efficient_weight_loading](08_memory_efficient_weight_loading) 包含一个额外的笔记本，展示如何通过PyTorch的`load_state_dict`方法更高效地加载模型权重
- [09_extending-tokenizers](09_extending-tokenizers) 包含GPT-2 BPE分词器的从零开始实现
- [10_llm-training-speed](10_llm-training-speed) 展示提高LLM训练速度的PyTorch性能技巧
- [11_qwen3](11_qwen3) Qwen3 0.6B和Qwen3 30B-A3B（专家混合模型）的从零开始实现，包括加载基础、推理和编码模型变体的预训练权重的代码

<br>
<br>

[![Link to the video](https://img.youtube.com/vi/Zar2TJv-sE0/0.jpg)](https://www.youtube.com/watch?v=Zar2TJv-sE0)
