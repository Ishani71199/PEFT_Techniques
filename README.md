# 🚀 PEFT for LLMs: Techniques and Comparative Insights
A detailed implementation of 3 popular PEFT (Parameter Efficient Fine-Tuning) techniques and their comparison. The main focus of this project is to understand the mechanics behind these methods.
## 🎯 3 Techniques (core idea):
-	**Simple Fine-Tuning:** Unfreeze and update only the final two layers (classification head).
-	**Adapter-Based Fine-Tuning:** Integrate lightweight adapter modules within each transformer block to enable efficient task-specific adaptation.
-	**LoRA (Low-Rank Adaptation):** Inject low-rank decomposition matrices into the model's weight updates, allowing parameter-efficient fine-tuning across the network.
  
To know more about these click on respective PEFT techniques: [Simple FT](https://github.com/Ishani71199/PEFT_Techniques/blob/main/Simple%20Fine%20Tuning.md/), [adapter FT](https://github.com/Ishani71199/PEFT_Techniques/blob/main/Adapter%20layer.md), [LoRA](https://github.com/Ishani71199/PEFT_Techniques/blob/main/LoRA.md)

## 📂 Project Structure:
**data_preprocessing.py:** Includes utility functions for dataset handling and preprocessing.
It performs data loading, text tokenization, and data loader configuration.

**modeling/:** Contains implementations of the three PEFT strategies:
-	Simple Fine-Tuning
-	Adapter Modules
-	LoRA (Low-Rank Adaptation)
  
**evaluation.py:** Provides helper functions for recording experiment outcomes and generating visualizations.

**finetuning.py:** The central script responsible for executing the fine-tuning experiments.

## 📦 Usage
Install the packages: pip install -r requirements.txt

Use a GPU for faster training. 

Run the project using the command: python finetuning.py

## 📝 Details
**PEFT Techniques:** Simple Fine-Tuning, Adapter Layers, LoRA

**Dataset:** [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb). This is a dataset for binary sentiment classification. (Binary Text Classification)

**Model:** [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased). DistilBERT is a smaller, faster, and lighter version of BERT (Bidirectional Encoder Representations from Transformers) developed by Hugging Face. It is useful for Text classification.

## 📈 Results and some Insights:
**1) Percentage of trainable parameters:**

![image](https://github.com/user-attachments/assets/73458646-7d5e-4ed3-82f4-b85fb17a9197)

All these techniques train less than 1% of the model’s parameters, making them highly parameter-efficient.

**2) Comparing Training Accuracy:**

![image](https://github.com/user-attachments/assets/060f57eb-ee03-482c-bc93-4d73d54fc3b4)

Adapter and LoRA achieved the best accuracies with 94.8938% and 93.0716% respectively.

**3) Comparing Training Time:**

![image](https://github.com/user-attachments/assets/bb8c9629-a084-4a02-a7ad-a6643cc37f55)

## 🤝 Reference:
[https://lightning.ai/pages/community/article/understanding-llama-adapters/](https://lightning.ai/pages/community/article/understanding-llama-adapters/)

[https://lightning.ai/pages/community/article/lora-llm/](https://lightning.ai/pages/community/article/lora-llm/)

[https://blog.gopenai.com/fine-tuning-llms-efficiently-9353d3b9a6d7](https://blog.gopenai.com/fine-tuning-llms-efficiently-9353d3b9a6d7)


