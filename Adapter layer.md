Adapter Layers are small neural network modules that are inserted inside the model, usually after each Transformer block. During fine-tuning, only the adapter layers are trained — the rest of the original model is frozen. So the model learns new tasks without forgetting old knowledge.

### Structure:
![image](https://github.com/user-attachments/assets/730126b2-fd2e-4e62-953b-9e4f42ca8724)

### Why Use a Bottleneck in Adapter Layers?
Forcing Compact Representations = Better Generalization

When you reduce the dimensionality, the model can't memorize or overfit to noise. It is forced to learn the essential task-specific features.

 - **Parameter Efficiency**
Suppose your model has 768 hidden units.
A full linear layer would have 768 x 768 = 589,824 weights.
But an adapter might reduce to 64:
•	Down projection: 768 x 64 = 49,152
•	Up projection: 64 x 768 = 49,152
•	Total = 98,304, which is ~6x fewer parameters!
After transformation, the adapter output is added back to the original hidden state. So if the **adapter is zero-initialized or disabled**, the model behaves exactly like the frozen base model. This makes adapters non-destructive and easy to plug in.

### Why Are Adapter Layers Zero-Initialized?

### 1. Stability in the Beginning
- If you initialize the adapter weights to zero, the initial output of the adapter is also zero. The adapter effectively does nothing initially.
- The full model output = base model output, untouched.
- This is especially important in PEFT (Parameter-Efficient Fine-Tuning), where the base model is frozen. You don’t want the adapter to harm its predictions from the start.
### 2. Gradual Learning
- The adapter starts from a non-intrusive state and slowly learns how to adjust the base model output during fine-tuning.
- This is very similar to residual learning — the adapter learns a correction, not a full rewrite.

### Do All Implementations Use Zero Init?
**Not necessarily.**
Some frameworks (e.g. Hugging Face's adapter-transformers) do use zero init. Others may use Xavier (Glorot) or normal distribution to avoid slow learning.
