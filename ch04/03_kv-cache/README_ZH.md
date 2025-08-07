# 额外内容：KV缓存

**本文件夹实现了向GPT模型添加KV缓存的功能。**

&nbsp;
## 概述

简而言之，KV缓存存储中间的键（K）和值（V）计算结果以便在推理过程中重复使用，这在生成响应时能带来显著的速度提升。缺点是它增加了代码的复杂性，增加了内存使用，并且不能在训练期间使用。然而，在部署LLM时，推理速度的提升通常值得在代码复杂性和内存方面做出权衡。

&nbsp;
## 工作原理

想象LLM正在生成一些文本。具体来说，假设LLM得到以下提示："Time flies"。

下图显示了使用第3章修改后的图形进行底层注意力分数计算的摘录，其中突出显示了键和值向量：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-1.png?3" width=800>

现在，正如我们在第2章和第4章中学到的，LLM一次生成一个单词（或token）。假设LLM生成了单词"fast"，使得下一轮的提示变成"Time flies fast"。这在下图中进行了说明：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-2.png?3" width=800>

正如我们通过比较前面的2个图所看到的，前两个token的键和值向量完全相同，在每一轮的下一个token文本生成中重新计算它们是浪费的。

因此，KV缓存的想法是实现一个缓存机制，存储先前生成的键和值向量以供重复使用，这有助于我们避免不必要的重新计算。

&nbsp;

## KV缓存实现

有许多方法可以实现KV缓存，主要思想是我们只为每个生成步骤中新生成的token计算键和值张量。

我选择了一个强调代码可读性的简单方法。我认为最简单的方法就是浏览代码更改来看看它是如何实现的。

这个文件夹中有两个文件：

1. [`gpt_ch04.py`](gpt_ch04.py)：取自第3章和第4章的独立代码，用于实现LLM并运行简单的文本生成函数
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py)：与上面相同，但进行了必要的更改以实现KV缓存。

您可以选择

a. 打开[`gpt_with_kv_cache.py`](gpt_with_kv_cache.py)文件并查找标记新更改的`# NEW`部分：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/new-sections.png?3" width=800>

b. 通过您选择的文件差异工具查看两个代码文件以比较更改：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/file-diff.png?3" width=800>

为了总结实现细节，这里是一个简短的演练。

&nbsp;

### 1. 注册缓存缓冲区

在`MultiHeadAttention`构造函数内部，我们添加两个非持久缓冲区`cache_k`和`cache_v`，它们将保存跨步骤连接的键和值：

```python
self.register_buffer("cache_k", None, persistent=False)
self.register_buffer("cache_v", None, persistent=False)
```

&nbsp;

### 2. 带有`use_cache`标志的前向传播

接下来，我们扩展`MultiHeadAttention`类的`forward`方法以接受`use_cache`参数。在将新的token块投影到`keys_new`、`values_new`和`queries`之后，我们要么初始化kv缓存，要么追加到我们的缓存：

```python
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
        
    # ...
    
    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```

&nbsp;


### 3. 清除缓存

在生成文本时，在独立序列之间（例如文本生成调用），我们必须重置两个缓冲区，因此我们还向`MultiHeadAttention`类添加了一个缓存重置方法：

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

&nbsp;

### 4. 在完整模型中传播`use_cache`

在对`MultiHeadAttention`类进行更改后，我们现在修改`GPTModel`类。首先，我们向构造函数添加token索引的位置跟踪：

```python
self.current_pos = 0
```

然后，我们用显式循环替换单行块调用，通过每个transformer块传递`use_cache`：

```python
def forward(self, in_idx, use_cache=False):
    # ...
 
    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,            
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )
    
    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds
    # ...
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

上述更改还需要对`TransformerBlock`类进行小的修改以接受`use_cache`参数：
```python
    def forward(self, x, use_cache=False):
        # ...
        self.att(x, use_cache=use_cache)
```

最后，我们向`GPTModel`添加模型级重置，以便一次清除所有块缓存：

```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

&nbsp;

### 5. 在生成中使用缓存

通过对`GPTModel`、`TransformerBlock`和`MultiHeadAttention`的更改，最后，这里是我们如何在简单文本生成函数中使用KV缓存：

```python
def generate_text_simple_cached(model, idx, max_new_tokens, 
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # 用完整提示初始化缓存
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) 选择具有最高对数概率的token（贪婪采样）
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) 将其追加到运行序列
                idx = torch.cat([idx, next_idx], dim=1)
                # c) 仅向模型提供新token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

注意，我们在c)中仅通过`logits = model(next_idx, use_cache=True)`向模型提供新token。没有缓存时，我们向模型提供整个输入`logits = model(idx[:, -ctx_len:], use_cache=False)`，因为它没有存储的键和值可以重复使用。

&nbsp;

## 简单性能比较

在概念层面介绍了KV缓存之后，最大的问题是它在一个小例子的实际应用中表现如何。为了尝试这个实现，我们可以将前面提到的两个代码文件作为Python脚本运行，这将运行小型124M参数LLM来生成200个新token（给定4个token的提示"Hello, I am"开始）：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

在配备M4芯片的Mac Mini（CPU）上，结果如下：

|                        | Tokens/sec |
| ---------------------- | ---------- |
| `gpt_ch04.py`          | 27         |
| `gpt_with_kv_cache.py` | 144        |

因此，正如我们所看到的，我们已经在小型124M参数模型和短200个token序列长度下获得了约5倍的速度提升。（注意，此实现针对代码可读性进行了优化，而不是针对CUDA或MPS运行时速度进行了优化，这需要预分配张量而不是重新实例化和连接它们。）

**注意：** 在两种情况下，模型都生成"胡言乱语"，即看起来像这样的文本：

> 输出文本：Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous bore ITVEGIN ministriesysics Kle functional recountrictionchangingVirgin embarrassedgl ...

这是因为我们还没有训练模型。下一章训练模型，您可以在训练好的模型上使用KV缓存（但是，KV缓存仅用于推理期间）来生成连贯的文本。在这里，我们使用未训练的模型来保持代码简单。

不过，更重要的是，`gpt_ch04.py`和`gpt_with_kv_cache.py`实现产生完全相同的文本。这告诉我们KV缓存实现是正确的——很容易犯索引错误，这可能导致不同的结果。

&nbsp;

## KV缓存的优缺点

随着序列长度的增加，KV缓存的好处和缺点在以下方面变得更加明显：

- [好处] **计算效率提高**：没有缓存时，步骤*t*的注意力必须将新查询与*t*个先前的键进行比较，因此累积工作呈二次方缩放，O(n²)。使用缓存，每个键和值计算一次然后重复使用，将每步总复杂度降低到线性，O(n)。

- [缺点] **内存使用线性增加**：每个新token都会追加到KV缓存。对于长序列和较大的LLM，累积的KV缓存变得更大，这可能消耗大量甚至禁止性的（GPU）内存。作为解决方法，我们可以截断KV缓存，但这增加了更多复杂性（但同样，在部署LLM时这可能是值得的。）

&nbsp;

## 优化KV缓存实现

虽然我上面的KV缓存概念实现有助于清晰度，主要面向代码可读性和教育目的，但在现实世界场景中部署它（特别是对于更大的模型和更长的序列长度）需要更仔细的优化。

&nbsp;
### 扩展缓存时的常见陷阱

- **内存碎片和重复分配**：如前所示，通过`torch.cat`连续连接张量会导致由于频繁的内存分配和重新分配而产生性能瓶颈。

- **内存使用的线性增长**：没有适当的处理，KV缓存大小对于非常长的序列变得不切实际。

&nbsp;
#### 技巧1：预分配内存

与其重复连接张量，我们可以根据预期的最大序列长度预分配足够大的张量。这确保了一致的内存使用并减少了开销。在伪代码中，这可能如下所示：

```python
# 键和值的预分配示例
max_seq_len = 1024  # 预期的最大序列长度
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
```

在推理期间，我们可以简单地写入这些预分配张量的切片。

&nbsp;
#### 技巧2：通过滑动窗口截断缓存

为了避免耗尽我们的GPU内存，我们可以实现带有动态截断的滑动窗口方法。通过滑动窗口，我们在缓存中仅维护最后`window_size`个token：

```python
# 滑动窗口缓存实现
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```

&nbsp;
#### 实践中的优化

您可以在[`gpt_with_kv_cache_optimized.py`](gpt_with_kv_cache_optimized.py)文件中找到这些优化。

在配备M4芯片的Mac Mini（CPU）上，使用200个token生成和等于上下文长度的窗口大小（以保证相同结果），代码运行时间比较如下：

|                                  | Tokens/sec |
| -------------------------------- | ---------- |
| `gpt_ch04.py`                    | 27         |
| `gpt_with_kv_cache.py`           | 144        |
| `gpt_with_kv_cache_optimized.py` | 166        |

不幸的是，在CUDA设备上速度优势消失了，因为这是一个微小的模型，设备传输和通信超过了KV缓存对这个小模型的好处。

&nbsp;
## 其他资源

1. [Qwen3从零开始的KV缓存基准测试](../../ch05/11_qwen3#pro-tip-2-speed-up-inference-with-compilation)
2. [Llama 3从零开始的KV缓存基准测试](../../ch05/07_gpt_to_llama/README.md#pro-tip-3-speed-up-inference-with-compilation)
3. [从零开始理解和编码LLM中的KV缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) -- 这个README的更详细写作
