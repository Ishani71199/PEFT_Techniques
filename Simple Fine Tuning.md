This approach represents conventional fine-tuning, wherein the pre-trained language model is partially frozen—specifically, all layers except the pre-classification and classification (task-specific) heads are kept static. The earlier transformer layers, which capture general-purpose linguistic and semantic representations, are frozen to preserve their pretrained knowledge and prevent overfitting on limited task-specific data. Conversely, the final layers near the output, which are more task-sensitive, are unfrozen and undergo weight updates during training. These layers are fine-tuned on the downstream dataset, enabling the model to adapt the generalized embeddings to the nuances of the specific task. The frozen layers function as static feature extractors, providing a robust foundation for task adaptation.           

 (![image](https://github.com/user-attachments/assets/f4e87600-3b91-4424-9949-2df694bef173)


But the main question behind this technique remains: How many layers to be unfreeze?
