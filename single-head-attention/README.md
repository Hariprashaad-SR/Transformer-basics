# SELF ATTENTION MECHANISM

An attention mechanism is an Encoder-Decoder kind of neural network architecture that allows the model to focus on specific sections of the input while executing a task. It dynamically assigns weights to different elements in the input, indicating their relative importance or relevance. By incorporating attention, the model can selectively attend to and process the most relevant information, capturing dependencies and relationships within the data. This mechanism is particularly valuable in tasks involving sequential or structured data, such as natural language processing or computer vision, as it enables the model to effectively handle long-range dependencies and improve performance by selectively attending to important features or contexts.

![](https://beehiiv-images-production.s3.amazonaws.com/uploads/asset/file/9674b20d-c1df-4b9c-8c40-b4f54da760ee/self_attention_thumbnail.png?t=1700825519)

### (i) Masking :
Masking is done in the decoder part where our tokens should not be able to look into future context(Look-ahead context). It can be done using
- Loops
- Matrix multiplication
- Softmax fn

### (ii) Scaling :
Scaling is done to make sure the the distribution of the attention score have nearly a normal distribution(mean = 0, var = 1)

Attention(Q,K,V) = ${\frac{Q.K.trans}{\sqrt{dk}}}.V$

## Sample pytorch code for attention mechanism
```
import math

def scaled_dot_prod(q, k, v, mask = False):
  d_k = q.size()[-1]
  scaled = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

  if mask:
    tril = torch.tril(torch.ones(scaled.size()))
    mask = tril.masked_fill(tril == 0, float('-inf'))
    mask = mask.masked_fill(tril == 1, 0)
    scaled += mask

  attention = F.softmax(scaled, dim = -1)
  values = attention @ v

  return values
```

Hence, after the attention mechanism, the output we return are more context aware than the input tokens
