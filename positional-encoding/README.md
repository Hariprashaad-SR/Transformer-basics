# Positional Encoding(UPDATING)
Positional encoding describes the location or position of an entity in a sequence so that each position is assigned a unique representation. There are many reasons why a single number, such as the index value, is not used to represent an item’s position in transformer models. For long sequences, the indices can grow large in magnitude. If you normalize the index value to lie between 0 and 1, it can create problems for variable length sequences as they would be normalized differently.

![](https://machinelearningmastery.com/wp-content/uploads/2022/01/PE3.png)

Positional encodings are crucial in Transformer models for several reasons:

- Preserving Sequence Order: Transformer models process tokens in parallel, lacking inherent knowledge of token order. Positional encodings provide the model with information about the position of tokens in the sequence, ensuring that the model can differentiate between tokens based on their position. This is essential for tasks where word order matters, such as language translation and text generation.

- Maintaining Contextual Information: In natural language processing tasks, the meaning of a word often depends on its position in the sentence. For example, in the sentence “The cat sat on the mat,” the word “cat” has a different meaning than in “The mat sat on the cat.” transformer

- Enhancing Generalization: By incorporating positional information, transformer models can generalize better across sequences of different lengths. This is particularly important for tasks where the length of the input sequence varies, such as document summarization or question answering. Positional encodings enable the model to handle input sequences of varying lengths without sacrificing performance.

- Mitigating Symmetry: Without positional encodings, the self-attention mechanism in Transformer models would treat tokens symmetrically, potentially leading to ambiguous representations. Positional encodings introduce an asymmetry into the model, ensuring that tokens at different positions are treated differently, thereby improving the model’s ability to capture long-range dependencies.

---

positional encoding_{(pos,2i)} => $sin({\frac{pos}{10000^{2_{i} / d_{model}}}})$

positional encoding_{(pos, 2i + 1)} => $cos({\frac{pos}{10000^{2_i - 1 / d_{model}}}})$

```
i => dimension index
dmodel => embedding length
pos => position of word in sequence

Reasons:
- Periodicity
- Constrained values
- Easy to extrapolate for long sequences
```
In summary, positional encodings are essential in Transformer models for preserving sequence order, maintaining contextual information, enhancing generalization, and mitigating symmetry. They enable Transformer models to effectively process and understand input sequences, leading to improved performance across a wide range of natural language processing tasks.
