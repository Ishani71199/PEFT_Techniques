Instead of updating all weights, it inserts trainable low-rank matrices (A and B) into the attention layers. The original pre-trained weights are frozen. This significantly reduces the number of trainable parameters while preserving performance.

![image](https://github.com/user-attachments/assets/f85d7f05-8422-4a1f-82c9-639c5d03e2c4)

![image](https://github.com/user-attachments/assets/d5b33f38-bb30-4662-a122-2de51d273bf9)
### Why 2 matrices (A & B) are used in LoRA instead of one?
Instead of directly learning a full-rank weight update ΔW (which would require a large number of parameters), LoRA approximates it using a low-rank decomposition:

![image](https://github.com/user-attachments/assets/1b432e72-3812-428f-bf4b-2c5f82e2aa3f)

where:

![image](https://github.com/user-attachments/assets/0027462f-0c42-4c63-ae8c-66009384454e)

### Benefits of Using Two Low-Rank Matrices
- **Parameter Efficiency:**
You only need to learn d × r + r × k parameters instead of d × k, which can be a huge saving for large models.
- **Preserve Pretrained Weights:**
The original weights W remain frozen. Only the lightweight WA and WB matrices are trained, making LoRA non-intrusive.
- **Computational Efficiency:**
The multiplication of two smaller matrices adds only a small overhead during training and inference.

