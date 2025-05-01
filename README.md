# PEFT for LLMs: Techniques and Comparative Insights
A detailed implementation of 3 popular PEFT (Parameter Efficient Fine-Tuning) techniques and their comparison. The main focus of this project is to understand the mechanics behind these methods.
### 3 techniques (core idea):
-	**Simple Fine-Tuning:** Unfreeze and update only the final two layers, including the classification head.
-	**Adapter-Based Fine-Tuning:** Integrate lightweight adapter modules within each transformer block to enable efficient task-specific adaptation.
-	**LoRA (Low-Rank Adaptation):** Inject low-rank decomposition matrices into the model's weight updates, allowing parameter-efficient fine-tuning across the network.
  
To know more about these click on respective PEFT techniques: [Simple FT](https://github.com/Ishani71199/PEFT_Techniques/blob/main/Simple%20Fine%20Tuning.md/), adapter FT, LoRA
