+++
date = '2025-11-17T12:42:22+08:00'
draft = false
title = 'Transformer'
+++

# Transformer

大语言模型要具有复杂理解以及生成人类语言的能力。大语言模型不是为特定的语言任务所设计而是具有更广泛的通用能力。大语言模型的成功归因于Transformer架构，以及用于训练的海量数据。

通过编写代码基于Transformer架构实现类似ChatGPT的大语言模型。

大语言模型的大既指参数量规模，又指海量数据。

Transformer很重要的一点是它能够选择性的关注输入的不同部分。关键组件是self-attention，能够衡量输入序列中每个token相对于其他token的相对重要性。

大语言模型的构建包含pretraining and fine-tuning两个阶段。pretraining阶段是在海量unlabeled text data上进行self-supervised learning，去学到一些general represention。fine-tuning是train on labeled data。

Transformer分为encoder和decoder部分。encoder负责将输入序列编码成a series of numerical representations or vectors that capture the contextual information of the input，解码器根据当前的these encoded vectors以及当前输入完成下一个词的预测。

实现预训练的代码、复用公开可用的预训练模型

The next-word prediction task是一种自监督学习，不需要为训练数据提供标签，他利用数据自身的结构。用文本中的下一个词作为要训练的标签。

Autoregressive models整合早先的输出作为未来预测的输入。

original transformer 的encoder和decoder重复6次。GPT-3有96层transformer layers 以及175b的parameters。

emergent behavior：是模型能够执行未被显示训练的任务的能力。

![image.png](/attachment/Transformer/image.png)

### ***Working with text data***

🍎the required steps for preparing the embeddings used by an LLM

splitting text into individual word and subword tokens

converting words into tokens

turning tokens into embedding vectors

🍎be encoded into vector representations

🍎advanced tokenization schemes like byte pair encoding

🍎implement a sampling and data-loading strategy to produce the input-output pairs

*2.1 Understanding word embeddings*

embedding：将data convert成vector represention。

an embedding is a mapping from discrete objects, such as words, images, or even entire documents, to points in a continuous vector space。

因此需要represent words as continuous-valued vectors。 

![image.png](/attachment/Transformer/image%201.png)

text embeding包含word embeding、embeddings for sentences, paragraphs, or whole documents。

Sentence or paragraph embeddings are popular choices for *retrieval-augmented generation。*

Word2Vec思想是相似上下文里的单词具有相似的语义，投影到向量空间时clustered together。

LLM会自己生成嵌入向量而不是用pretrained models such as Word2Vec。

The smallest GPT-2 models (117M and 125M parameters)：an embedding size of 768 dimensions。

The largest GPT-3 model (175B parameters)：an embedding size of 12,288 dimensions

2.2 Tokenizing text

spliting input text into individual tokens

![image.png](/attachment/Transformer/image%202.png)

```python
# use the re,split command with the following syntax to split a text on whitespace charaters
import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)
```

The result is a list of individual words,whitespaces,and punctuation characters

`['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test.']`

```python
# modify the regular expression splits on whitespaces (\s), commas, and periods ([,.])
result = re.split(r'([,.]|\s)', text)
print(result)
```

the words and punctuation characters are now separate list entries

`['Hello', ',', '', ' ', 'world', '.', '', ' ', 'This', ',', '', ' ', 'is',' ', 'a', ' ', 'test', '.', '']`

```python
# remove these redundant characters
# strip() 是字符串（str）对象的一个方法，用来去掉字符串两端的指定字符（默认是空白符，包括空格、换行符\n、制表符\t 等）
result = [item for item in result if item.strip()]
print(result)
```

whitespace-free output

`['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']`

Removing whitespaces reduces the memory and computing requirements. However, keeping whitespaces can be useful if we train models that are sensitive to the exact structure of the text (for example,Python code, which is sensitive to indentation and spacing).

```python
# handle other types of punctuation, such as question marks, quotation marks, and the double-dashes
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)
```

`['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']`

*2.3 Converting tokens into token IDs*

convert these tokens from a Python string to an integer representation to produce the token IDs。

build a vocabulary。This vocabulary defines how we map each unique word and special character to a unique integer。

![image.png](/attachment/Transformer/image%203.png)

```python
# create a list of all unique tokens and sort them alphabetically
all_words = sorted(set(preprocessed))

# create the vocabulary which defines how we map each unique word and special character to a unique integer
vocab = {token:integer for integer,token in enumerate(all_words)}
```

apply this vocabulary to convert new text into token IDs and turn token IDs into text。

![image.png](/attachment/Transformer/image%204.png)

```python
# implement a complete tokenizer class
# with an encode method that splits text into tokens 
# and carries out the string-to-integer mapping to produce tokenIDs via the vocabulary
# a decode method that carries out the reverse integer-to-string mapping to convert the token IDs back into text.
class SimpleTokenizerV1:
	def __init__(self, vocab):
		self.str_to_int = vocab 
		self.int_to_str = {i:s for s,i in vocab.items()} 
		# .items() return all key-value pairs of dict.such as dict_items([('hello', 1), ('world', 2)])

	def encode(self, text): 
		preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
		preprocessed = [
		item.strip() for item in preprocessed if item.strip()]
		# if后的strip是判断有没有空内容，比如全空格或者换行符，判断false过滤掉
		# 最前面的strip是如果不是全空内容，就对内容进行清洗去掉空格换行符保留游泳内容
		# 没有if后的strip前面的strip能不能清洗掉全空的内容,不行。“    ”.strip()会变成“”
		ids = [self.str_to_int[s] for s in preprocessed]
		return ids

	def decode(self, ids): 
		text = " ".join([self.int_to_str[i] for i in ids]) 

		text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) 
		# 在字符串 string 中，查找所有符合 pattern 的子串，并用 repl 替换掉。
		# 在正则表达式替换（re.sub）里，\1 表示：引用第 1 个括号里捕获的内容。
		# \s+ → 一个或多个空白字符（空格、换行、制表符）
		# ([,.?!"()']) → 捕获括号内的任意一个标点符号
		# \1 → 正则中第一个括号捕获的内容（即标点符号本身）
		# 作用就是：去掉标点前面的多余空格。
		
		
		return text
		

"""
In [18]: a = ['Hello', 'world', ' ! ', '     ']

In [19]: [item.strip() for item in a]
Out[19]: ['Hello', 'world', '!', '']

In [20]: [item.strip() for item in a if item.strip()]
Out[20]: ['Hello', 'world', '!']

In [21]: [item for item in a if item.strip()]
Out[21]: ['Hello', 'world', ' ! ']

"""
```

Using the SimpleTokenizerV1 Python class, we can now instantiate new tokenizer objects via an existing vocabulary, which we can then use to encode and decode text

![image.png](/attachment/Transformer/image%205.png)

```python
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," 
 Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))
```

*2.4 Adding special context tokens*

modify the tokenizer to handle unknown words and address the usage and addition of special context tokens。

special tokens including markers for unknown words and document boundaries, <|unk|> and <|endoftext|>

![image.png](/attachment/Transformer/image%206.png)

```python
# add <unk> and <|endoftext|> to list of all unique words.

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
```

A simple text tokenizer that handles unknown words

```python
class SimpleTokenizerV2:
	def __init__(self, vocab):
		self.str_to_int = vocab
		self.int_to_str = { i:s for s,i in vocab.items()}
	
	def encode(self, text):
		preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
		preprocessed = [
		item.strip() for item in preprocessed if item.strip()
		]
		preprocessed = [item if item in self.str_to_int 
		else "<|unk|>" for item in preprocessed]
		ids = [self.str_to_int[s] for s in preprocessed]
		return ids
	
	def decode(self, ids):
		text = " ".join([self.int_to_str[i] for i in ids])
		text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text) 
		return text
		
```

*2.5 Byte pair encoding*

Python open source library called *tiktoken* (https://github.com/openai/tiktoken), which implements the BPE algorithm very efficiently based on source code in Rust.

`pip install tiktoken`

The code we will use is based on tiktoken 0.7.0.

```python
from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

# instantiate the BPE tokenizer from tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# an encode method:
text = (
 "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
 "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

# the decode method
strings = tokenizer.decode(integers)
print(strings)
```

the BPE tokenizer has a total vocabulary size of 50,257

BPE breaks down words that aren’t in its predefined vocabulary into smaller subword units or even individual characters, enabling it to handle out-of-vocabulary words.

*2.6 Data sampling with a sliding window*

*2.7 Creating token embeddings*

convert the token IDs into embedding vectors

![image.png](/attachment/Transformer/image%207.png)

how the token ID to embedding vector conversion

```python
# the embedding layer is essentially a lookup operation 
# that retrieves rows from the embedding layer’s weight matrix via a token ID.
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

# apply embedding layer to a token ID to obtain the embedding vector
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(torch.tensor([2, 3, 5, 1])))

# Each row in this output matrix is obtained via a lookup operation from the embedding weight matrix

```

*2.8 Encoding word positions*

it is helpful to inject additional position information into the LLM

two broad categories of position-aware embeddings: relative positional embeddings and absolute positional embeddings

OpenAI’s GPT models use absolute positional embeddings that are optimized during the training process rather than being fixed or predefined like the positional encodings in the original transformer model.

```python
# token embedding
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# position embedding
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

# add pos_embeddings to token_embeddings
input_embeddings = token_embeddings + pos_embeddings
```

![image.png](/attachment/Transformer/image%208.png)

### *Coding attention mechanisms*

🍎simplified attention mechanism

🍎add a causal attention mask to prevent the LLM from accessing future tokens

🍎add a dropout mask to reduce overfitting in LLMs

🍎multi-head attention: multiple instances of causal attention

🍎creating multi-head attention modules involves batched matrix multiplications

*3.4.2 Implementing a compact self-attention Python class*

1. initializes trainable weight matrices (W_query, W_key, and W_value)
2. compute the attention scores (attn_scores) by multiplying queries and keys
3. normalizing these scores using softmax to get attn_weights
4. create a context vector by weighting the values with these attn_weights

$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$

a significant advantage of using nn.Linear instead of manually implementing nn.Parameter(torch.rand(...)) is that nn.Linear has an optimized weight initialization scheme, contributing to more stable and effective model training.

```python

class SelfAttention_v2(nn.Module):
	def __init__(self, d_in, d_out, qkv_bias=False):
		super().__init__()
		
		self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
	def forward(self, x): # shape of x: (seq_len, d_in)
		keys = self.W_key(x) # shape of keys: (seq_len, d_out)
		queries = self.W_query(x) # shape of keys: (seq_len, d_out)
		values = self.W_value(x) # shape of keys: (seq_len, d_out)
		attn_scores = queries @ keys.T # (seq_len, seq_len)
		attn_weights = torch.softmax(
		attn_scores / keys.shape[-1]**0.5, dim=-1
		) # keys.shape[-1]: d_out
		# torch.softmax(......, dim=-1),按最后一维算softmax，也就是按行
		context_vec = attn_weights @ values
		return context_vec
		
	
# inputs contains six embedding vectors
# results contains six context vectors	
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
```

![image.png](/attachment/Transformer/image%209.png)

*3.5 Hiding future words with causal attention*

Causal attention, also known as masked attention 让模型只关注sequence中的previous 和 current input

mask out the future tokens

![image.png](/attachment/Transformer/image%2010.png)

dropout in the attention mechanism is typically applied at two specific times: after calculating the attention weights or after applying the attention weights to the value vectors.

apply the dropout mask after computing the attention weights更常见。

![image.png](/attachment/Transformer/image%2011.png)

```python
class CausalAttention(nn.Module):
	def __init__(self, d_in, d_out, context_length,
		dropout, qkv_bias=False):
		super().__init__()
		self.d_out = d_out
		self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.dropout = nn.Dropout(dropout) 
		self.register_buffer(
		'mask',
		torch.triu(torch.ones(context_length, context_length),
		diagonal=1)
		) 
	def forward(self, x):
		b, num_tokens, d_in = x.shape 
		
		keys = self.W_key(x)
		queries = self.W_query(x)
		values = self.W_value(x)
		
		attn_scores = queries @ keys.transpose(1, 2) 
		# creating a mask with 1s above the diagonal 
		# and then replacing these 1s with negative infinity (-inf) values
		attn_scores.masked_fill_( 
		self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) 
		
		attn_weights = torch.softmax(
		attn_scores / keys.shape[-1]**0.5, dim=-1
		)
		
		attn_weights = self.dropout(attn_weights)
		context_vec = attn_weights @ values
		return context_vec
		
		
		
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
```

*3.6 Extending single-head attention to multi-head attention*

processing the heads in sequential

multiple heads are implemented by creating a list of CausalAttention objects (self.heads)

![image.png](/attachment/Transformer/image%2012.png)

```python
class MultiHeadAttentionWrapper(nn.Module):
	def __init__(self, d_in, d_out, context_length,
		dropout, num_heads, qkv_bias=False):
		super().__init__()
		self.heads = nn.ModuleList(
		[CausalAttention(
		d_in, d_out, context_length, dropout, qkv_bias
		) 
		for _ in range(num_heads)]
		)
	def forward(self, x):
		return torch.cat([head(x) for head in self.heads], dim=-1)
```

processing the heads in parallel

splits the input into multiple heads by reshaping the projected query, key, and value
tensors and then combines the results from these heads after computing attention

split the d_out dimension into num_heads and head_dim, where head_dim = d_out / num_heads.

This splitting is then achieved using the .view method: a tensor of dimensions (b, num_tokens, d_out) is reshaped to dimension (b, num_tokens, num_heads, head_dim)

![image.png](/attachment/Transformer/image%2013.png)

```python
class MultiHeadAttention(nn.Module):
	def __init__(self, d_in, d_out, 
		context_length, dropout, num_heads, qkv_bias=False):
		super().__init__()
		assert (d_out % num_heads == 0), \
		"d_out must be divisible by num_heads"
		self.d_out = d_out
		self.num_heads = num_heads
		self.head_dim = d_out // num_heads 
		self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.out_proj = nn.Linear(d_out, d_out) 
		self.dropout = nn.Dropout(dropout)
		self.register_buffer(
		"mask",
		torch.triu(torch.ones(context_length, context_length),
		diagonal=1)
		)
	def forward(self, x):
		b, num_tokens, d_in = x.shape
		keys = self.W_key(x) 
		queries = self.W_query(x) 
		values = self.W_value(x) 
		keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
		values = values.view(b, num_tokens, self.num_heads, self.head_dim) 
		queries = queries.view( 
		b, num_tokens, self.num_heads, self.head_dim 
		) 
		keys = keys.transpose(1, 2) 
		queries = queries.transpose(1, 2) 
		values = values.transpose(1, 2) 
		attn_scores = queries @ keys.transpose(2, 3) 
		mask_bool = self.mask.bool()[:num_tokens, :num_tokens] 
		
		attn_scores.masked_fill_(mask_bool, -torch.inf) 
		attn_weights = torch.softmax(
		attn_scores / keys.shape[-1]**0.5, dim=-1)
		attn_weights = self.dropout(attn_weights)
		context_vec = (attn_weights @ values).transpose(1, 2) 
		
		context_vec = context_vec.contiguous().view(
		b, num_tokens, self.d_out
		)
		context_vec = self.out_proj(context_vec) 
		return context_vec

		
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```

### *Implementing a GPT model from scratch to generate text*

*This chapter covers*

- Coding a GPT-like large language model (LLM) that can be trained to generate human-like text
- Normalizing layer activations to stabilize neural network training
- Adding shortcut connections in deep neural networks
- Implementing transformer blocks to create GPT models of various sizes
- Computing the number of parameters and storage requirements of GPT models

*4.1 Coding an LLM architecture*

parameters指的是trainable weights of the model

a GPT placeholder architecture (DummyGPTModel)

the order in which we tackle the individual concepts required to code the final GPT architecture

先code出一个GPT placeholder architecture calling DummyGPTModel，然后得到the individual core pieces ，最终assembling起来。

一个DummyGPTModel包括token embeddings , positional embedding , dropout , 一系列的transformer blocks(DummyTransformerBlock) 最后一个Layer Normalization(DummyLayerNorm)和一个Linear output layer

![image.png](/attachment/Transformer/image%2014.png)

```python
GPT_CONFIG_124M = {
	"vocab_size": 50257, # Vocabulary size
	"context_length": 1024, # model能够处理的最大token和positional embedding数量
	"emb_dim": 768, # 每个Token的Embedding size
	"n_heads": 12, # Number of attention heads
	"n_layers": 12, # Transformer Block的数量
	"drop_rate": 0.1, # Dropout rate
	"qkv_bias": False # Query-Key-Value bias
}
# A placeholder GPT model architecture class
import torch
import torch.nn as nn
	class DummyGPTModel(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
		self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
		self.drop_emb = nn.Dropout(cfg["drop_rate"])
		self.trf_blocks = nn.Sequential( 
		*[DummyTransformerBlock(cfg) 
		for _ in range(cfg["n_layers"])] 
		) 
		# *起unpack的作用，具体是将列表里的元素给一一拿出来。
		self.final_norm = DummyLayerNorm(cfg["emb_dim"]) 
		self.out_head = nn.Linear(
		cfg["emb_dim"], cfg["vocab_size"], bias=False
		)
	def forward(self, in_idx):
		batch_size, seq_len = in_idx.shape
		tok_embeds = self.tok_emb(in_idx)
		pos_embeds = self.pos_emb(
		torch.arange(seq_len, device=in_idx.device)
		)
		x = tok_embeds + pos_embeds
		x = self.drop_emb(x)
		x = self.trf_blocks(x)
		x = self.final_norm(x)
		logits = self.out_head(x)
		return logits
		
class DummyTransformerBlock(nn.Module): 
	def __init__(self, cfg):
		super().__init__()
	def forward(self, x): 
		return x
class DummyLayerNorm(nn.Module): 
	def __init__(self, normalized_shape, eps=1e-5): 
		super().__init__()
	def forward(self, x):
		return x
```

*4.2 Normalizing activations with layer normalization*

为什么用batch normalization而不是layer normalization？

训练有许多layers的deep neural network时会遇到vanishing or exploding gradients的问题。

实现Layer normalization可以提升训练的stability和efficiency，

layer normalization一般是放在multi-head attention module的前后

dim=-1表示向量的最后一维，对于一个two dimensional tensor来说也就是向量的columns。对于一个三维向量[bs, seq_len, embedding_size]来说最后一维就是每个token的embeddin

_size.

layer normalization的操作：out_norm = (out - mean) / torch.sqrt(var).

layer normalization一般是在输入tensor的last dimension操作的，这代表embedding dimension (emb_dim)。

```python
	class LayerNorm(nn.Module):
	def __init__(self, emb_dim):
		super().__init__()
		self.eps = 1e-5
		self.scale = nn.Parameter(torch.ones(emb_dim))
		self.shift = nn.Parameter(torch.zeros(emb_dim))
	def forward(self, x):
		mean = x.mean(dim=-1, keepdim=True)
		# 采用Biased variance有偏估计来求样本方差，因为维度768足够大，区别不大
		var = x.var(dim=-1, keepdim=True, unbiased=False)
		norm_x = (x - mean) / torch.sqrt(var + self.eps)
		# eps是极小值，防止division by zero的情况
		return self.scale * norm_x + self.shift
		# scale 和 shift是两个trainable parameters,初始分别是1，0，
		# 训练过程中如果被判定调整他能提升performance，模型会自动的调整他。
```

方差与样本方差，以及样本方差的有偏估计与无偏估计。

方差是除n，为什么样本方差除n反而是有偏估计，样本方差无偏估计是除以n-1？因为采样过程中会低估方差，所以要通过贝塞尔修正来修正方差。方差是反应数据离散程度的，从所有的数据中采样一些样本，用样本的样本方差来反应总体方差的一个问题就是采样样本的过程肯定是概率越大的越容易采样，就导致采样的样本离散程度更集中，当然如果采样的数目n足够大，采样的样本的分布就无限分布总体的分布了。以如下正太分布为例。

![image.png](/attachment/Transformer/image%2015.png)

*4.3 Implementing a feed forward network with GELU activations*

为什么用GELU（鸡路）激活函数而不是relu激活函数？因为relu激活函数在0的地方不可微。

dead neurons：当输入小于0时输出永远是0对学习没有什么贡献，所以叫dead neurons。

GELU(x) = x⋅Φ(x), where Φ(x) is the cumulative distribution function of the standard Gaussian distribution。

standard Gaussian distribution

![image.png](/attachment/Transformer/image%2016.png)

standard Gaussian distribution的CDF

![image.png](/attachment/Transformer/image%2017.png)

(the original GPT-2 model was also trained with this approximation, which was found via curve fitting）

![image.png](/attachment/Transformer/image%2018.png)

m = nn.GELU()

An implementation of the GELU activation function

```python
class GELU(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, x):
		return 0.5 * x * (1 + torch.tanh(
		torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
		(x + 0.044715 * torch.pow(x, 3))
		))
```

A feed forward neural network module

```python
class FeedForward(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.layers = nn.Sequential(
		nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
		GELU(),
		nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
		)
	def forward(self, x):
		return self.layers(x)
```

![image.png](/attachment/Transformer/image%2019.png)

*4.4 Adding shortcut connections*

short connections和residual connection是一回事，用来解决vanishing gredient problem。

*vanishing gradient problem* 是指反向传播过程中梯度过小导致学习停滞不前，convergence delay。

不加Skip Connection 反向传播过程中梯度会越来越小。skip connection可以为梯度创建一个另外的更短的路径，通过skipping one or more layers让gradient flow

![image.png](/attachment/Transformer/image%2020.png)

![Visualizing the Loss Landscape of Neural Nets](/attachment/Transformer/image%2021.png)

Visualizing the Loss Landscape of Neural Nets

不加Skip Connection有很多的局部最小值，会导致优化困难

```python
class ExampleDeepNeuralNetwork(nn.Module):
	def __init__(self, layer_sizes, use_shortcut):
		super().__init__()
		self.use_shortcut = use_shortcut
		self.layers = nn.ModuleList([
		nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), 
		GELU()),
		nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), 
		GELU()),
		nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), 
		GELU()),
		nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), 
		GELU()),
		nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), 
		GELU())
		])
	def forward(self, x):
		for layer in self.layers:
			layer_output = layer(x) 
			if self.use_shortcut and x.shape == layer_output.shape: 
				x = x + layer_output
			else:
				x = layer_output
		return x
```

a function that computes the gradients

```python
def print_gradients(model, x):
	output = model(x) 
	target = torch.tensor([[0.]])
	loss = nn.MSELoss()
	loss = loss(output, target) 
	loss.backward()
	for name, param in model.named_parameters():
		if 'weight' in name:
		print(f"{name} has gradient mean of {param.grad.abs().mean().item()}") 
```

*4.5 Connecting attention and linear layers in a transformer block*

implement the *transformer block 包括* multi-head attention, layer normalization, dropout, feed forward layers, and GELU activations。

TransformerBlock的核心是包括 a multi-head attention mechanism (MultiHeadAttention) and a feed forward network (FeedForward)。其中Layer normalization (LayerNorm)是在以上两部分的前面，dropout是在以上两部分的后面。layer normalization在前面是叫Pre-LayerNorm，在原论文中LayerNorm是在后面叫Post-LayerNorm

![image.png](/attachment/Transformer/image%2022.png)

```python
class TransformerBlock(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		
		self.att = MultiHeadAttention(
			d_in=cfg["emb_dim"],
			d_out=cfg["emb_dim"],
			context_length=cfg["context_length"],
			num_heads=cfg["n_heads"], 
			dropout=cfg["drop_rate"],
			qkv_bias=cfg["qkv_bias"])
		self.ff = FeedForward(cfg)
		self.norm1 = LayerNorm(cfg["emb_dim"])
		self.norm2 = LayerNorm(cfg["emb_dim"])
		self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
		
	def forward(self, x):
	
		shortcut = x
		x = self.norm1(x)
		x = self.att(x)
		x = self.drop_shortcut(x)
		x = x + shortcut 
		shortcut = x 
		x = self.norm2(x)
		x = self.ff(x)
		x = self.drop_shortcut(x)
		x = x + shortcut 
		return x
```

*4.6 Coding the GPT model*

![image.png](/attachment/Transformer/image%2023.png)

```python
class GPTModel(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
		self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
		self.drop_emb = nn.Dropout(cfg["drop_rate"])
		
		self.trf_blocks = nn.Sequential(
		*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
		
		self.final_norm = LayerNorm(cfg["emb_dim"])
		self.out_head = nn.Linear(
		cfg["emb_dim"], cfg["vocab_size"], bias=False
		)
	def forward(self, in_idx):
		batch_size, seq_len = in_idx.shape
		tok_embeds = self.tok_emb(in_idx)
		
		pos_embeds = self.pos_emb(
		torch.arange(seq_len, device=in_idx.device)
		)
		x = tok_embeds + pos_embeds
		x = self.drop_emb(x)
		x = self.trf_blocks(x)
		x = self.final_norm(x)
		logits = self.out_head(x)
		return logits
```

```python
# 计算model的parameters数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# 计算model的parameters占的内存大小
# Calculates the total size in bytes (assuming float32, 4 bytes per parameter)
total_size_bytes = total_params * 4 
# Converts to megabytes
total_size_mb = total_size_bytes / (1024 * 1024) 
print(f"Total size of the model: {total_size_mb:.2f} MB")
```

以上方式计算的总参数量是163,009,536，实际GPT-2的参数量是124,412,160。原因是GPT-2里有个参数绑定（weight tying）的技术。具体做法就是在output layer里复用了the token embedding layer的权重。为什么要这样做了，因为这两个层的维度是vocabulary size50, 257非常的巨大。

*4.7 Generating text*

*greedy decoding: 取概率最大的位置处的token。*

用softmax function去将logits转化为概率分布，用torch.argmax选出概率最大处的索引。

![image.png](/attachment/Transformer/image%2024.png)

```python
def generate_text_simple(model, idx, max_new_tokens, context_size): 
	'''
		max_new_tokens:希望生成的最大token数量。
		context_size:模型能够处理的最大上下文长度。
	'''
	# 循环max_new_tokens次，每次生成一个新的token
	 for _ in range(max_new_tokens):
		 # 受限于模型能处理的上下文长度context_size,选最新的context_size个token。
		 idx_cond = idx[:, -context_size:] 
		 with torch.no_grad():
			 logits = model(idx_cond)
		 # 选出模型generate的token: (bs, n_tokens, vocab_size) ---> (bs, vocab_size)
		 logits = logits[:, -1, :] 
		 probas = torch.softmax(logits, dim=-1) # logits ---> Probability distribution
		 idx_next = torch.argmax(probas, dim=-1, keepdim=True) # (bs,1)
		 idx = torch.cat((idx, idx_next), dim=1) 
```

### *Pretraining on unlabeled data*

*5.1.1 Using GPT to generate text*

```python
import tiktoken
from chapter04 import generate_text_simple

def text_to_token_ids(text, tokenizer):
	encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
	encoded_tensor = torch.tensor(encoded).unsqueeze(0) 
	# .unsqueeze(0) adds the batch dimension
	return encoded_tensor
def token_ids_to_text(token_ids, tokenizer):
	flat = token_ids.squeeze(0) 
	# Removes batch dimension 
	return tokenizer.decode(flat.tolist())
	
start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
	model=model,
	idx=text_to_token_ids(start_context, tokenizer),
	max_new_tokens=10,
	context_size=GPT_CONFIG_124M["context_length"]
	)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

*5.1.2 Calculating the text generation loss*

not just generating next token but also measuring the quality of the generated token