# Encoder Architecture

![](https://kikaben.com/transformers-encoder-decoder/images/encoder-layer-norm.png)
### Understanding Transformers — Encoder
Encoders in Transformers, also known as the transformer encoder, are a key component of the transformer encoder-decoder architecture. They are responsible for analyzing and representing the input sequence in a way the model can understand. The encoder processes the input sequence and produces a continuous representation, or embedding, of the input. These embeddings are then passed to the decoder to generate the output sequence.

The transformer encoder architecture typically consists of multiple layers, each of which includes a self-attention mechanism and a feed-forward neural network. The self-attention mechanism allows the model to weigh the importance of different input sequence parts by calculating the embeddings' dot product. This mechanism is also known as multi-head attention.

The feed-forward network allows the model to extract higher-level features from the input. This network usually comprises two linear layers with a ReLU activation function in between. The feed-forward network allows the model to extract deeper meaning from the input data and more compactly and usefully represent the input.

**(i) Embeddings:**
- Raw text is converted into embeddings, which encode information into vectors.
- Positional embeddings are added to regular embeddings to encode word positions, ensuring the model understands the order of words.

**(ii) Self-Attention:** 
- Self-attention layers consider the context of words within a sentence, addressing issues like homonyms.
- It uses three tensors: Query, Key, and Value. Queries from the decoder are matched with Keys from the encoder to generate similarity scores, which are then used to weight the Values.

Reference : [Single-head attention](https://github.com/Hariprashaad-SR/Transformer-basics/tree/main/single-head-attention)

**(iii) Multi-Head Attention:**
- Multi-Head Attention consists of several self-attention units, each focusing on different text features, akin to using multiple filters in convolutional layers.
- Outputs from attention heads are concatenated and linearly transformed to prepare for the next layer.

Reference : [Multi-head attention](https://github.com/Hariprashaad-SR/Transformer-basics/tree/main/multihead-attention)

**(iv) Feed-Forward Layer:**
- Composed of two linear layers, a RELU activation, and a dropout layer, this layer processes embeddings independently and is critical for memorizing information.
- It is typically scaled up in larger transformer models.

**(v) Layer Normalization**
- Applies normalization across the features (dimensions) for each data sample.
- Used to stabilize and accelerate the training process by reducing the internal covariate shift.
- Helps in maintaining the mean and variance of activations close to zero and one, respectively.
  
Reference : [Layer Normalization](https://github.com/Hariprashaad-SR/Transformer-basics/blob/main/layer-normalization)

## **Transformer Encoder Layer:**
- Combines all the necessary layers into a single model.
- Two types of normalization can be applied: Post-layer normalization (original architecture) and Pre-layer normalization (used more frequently due to stability in training).
- The encoder generates embeddings from input tokens and passes them through stacked encoder layers.

