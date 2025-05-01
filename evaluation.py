"""This class is used to evaluate the results of the PEFT techniques
It saves the results to a JSON file and plots the results
"""

import matplotlib.pyplot as plt
import os
import json

class EvaluationUtils:
    def __init__(
        self,
        dir_path: str = "outputs/",
        test_losses: dict = None,
        test_accuracies: dict = None,
        train_losses: dict = None,
        train_accuracies: dict = None,
        training_times: dict = None,
        trainable_params: dict = None,
    ) -> None:
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)
        self.test_losses = test_losses
        self.test_accuracies = test_accuracies
        self.train_losses = train_losses
        self.train_accuracies = train_accuracies
        self.training_times = training_times
        self.trainable_params = trainable_params
    
    def __save_results(self):
        results = {
            "test_losses": self.test_losses,
            "test_accuracies": self.test_accuracies,
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "training_times": self.training_times,
            "trainable_params": self.trainable_params,
        }
        with open(f"{self.dir_path}/results.json", "w") as f:
            json.dump(results, f, indent=4)
        print("Results saved to results.json.")
    
    def __plot_training_curves(self):
        """
        Makes 2 bar plots:
        one for loss and one for accuracy
        """
        # Plot the training loss
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.train_losses.keys(), self.train_losses.values(), color="skyblue")
        plt.bar_label(bars)  
        plt.xlabel("PEFT techniques")
        plt.ylabel("Training Loss")
        plt.title("LLM PEFT: Best Training Loss comparison")
        plt.savefig(f"{self.dir_path}/training_loss.png")
        print("Training loss plot saved.")

        # Plot the training accuracy
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.train_accuracies.keys(), self.train_accuracies.values(), color="orange")
        plt.bar_label(bars) 
        plt.xlabel("PEFT techniques")
        plt.ylabel("Training Accuracy")
        plt.title("LLM PEFT: Best Training Accuracy comparison")
        plt.savefig(f"{self.dir_path}/training_accuracy.png")
        print("Training accuracy plot saved.")

    def __plot_test_curves(self):
        """
        Method to plot the test curves
        Makes 2 bar plots: one for loss and one for accuracy
        """
        # Plot the test loss
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.test_losses.keys(), self.test_losses.values(), color="skyblue")
        plt.bar_label(bars)  
        plt.xlabel("PEFT techniques")
        plt.ylabel("Test Loss")
        plt.title("LLM PEFT: Test Loss comparison")
        plt.savefig(f"{self.dir_path}/test_loss.png")
        print("Test loss plot saved.")

        # Plot the test accuracy
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.test_accuracies.keys(), self.test_accuracies.values(), color="orange")
        plt.bar_label(bars)  
        plt.xlabel("PEFT techniques")
        plt.ylabel("Test Accuracy")
        plt.title("LLM PEFT: Test Accuracy comparison")
        plt.savefig(f"{self.dir_path}/test_accuracy.png")
        print("Test accuracy plot saved.")

    def __plot_training_times(self):
        """
        Method to plot the training times
        Makes a bar plot and saves it
        """
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.training_times.keys(), self.training_times.values(), color="green")
        plt.bar_label(bars)  
        plt.xlabel("PEFT techniques")
        plt.ylabel("Training Time (hours)")
        plt.title("LLM PEFT: Training Time comparison")
        plt.savefig(f"{self.dir_path}/training_times.png")
        print("Training times plot saved.")

    def __plot_trainable_params(self):
        """
        Method to plot the % of trainable parameters
        Makes a bar plot and saves it
        """
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.trainable_params.keys(), self.trainable_params.values(), color="red")
        plt.bar_label(bars)  
        plt.xlabel("PEFT techniques")
        plt.ylabel("% of Trainable Parameters")
        plt.title("LLM PEFT: Trainable Parameters % comparison")
        plt.savefig(f"{self.dir_path}/trainable_params.png")
        print("Trainable params plot saved.")

    def compute_results(self):
        """
        Main method to compute the results
        """
        self.__save_results()
        self.__plot_training_curves()
        self.__plot_test_curves()
        self.__plot_training_times()
        self.__plot_trainable_params()
