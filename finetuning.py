"""
Script that fine-tunes a pre-trained large language model using multiple strategies"""

from rich import print
import time
from data_preprocessing import DatasetUtils
from modeling import BaseModel, SimpleFTModel, AdaptedModel, LoRAModel
from evaluation import EvaluationUtils
import traceback

# Global variables
MODEL_NAME = "distilbert/distilbert-base-uncased"
DATASET_NAME = "stanfordnlp/imdb"
NUM_EPOCHS = 25

MODEL_MAP = {
    "Baseline": {
        "model": BaseModel(model_uri=MODEL_NAME, num_classes=2, freeze_all=True),
    },
    "Simple": {"model": SimpleFTModel(), "initial_lr": 8e-4},
    "Adapter": {"model": AdaptedModel(bottleneck_dim=16), "initial_lr": 8e-4},
    "LoRA": {
        "model": LoRAModel(lora_rank=16, lora_alpha=16),
        "initial_lr": 2e-5,
    },
}

test_losses, test_accuracies = {}, {}
train_losses, train_accuracies = {}, {}
training_times, trainable_params = {}, {}

def get_time_in_hours(seconds: int) -> float:
    return seconds / 3600

def run_model(model_name, train_loader, val_loader, test_loader) -> None:
    # Get model and initial learning rate
    model = MODEL_MAP[model_name].get("model")
    initial_lr = MODEL_MAP[model_name].get("initial_lr")

    if model_name != "Baseline":
        print(f"Fine-tuning {model_name} model...\n")
        # 1. Get trainable parameter count
        trainable_params_percentage = model.get_parameter_count()
        # Add to dictionary
        trainable_params[model_name] = trainable_params_percentage
        # 2. Start timer
        start = time.time()
        # 3. Train the model
        train_loss, train_acc = model.train(
            train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate=initial_lr
        )
        # 4. End timer and calculate training time
        training_time = get_time_in_hours(time.time() - start)
        # Add to dictionary
        train_losses[model_name] = train_loss
        train_accuracies[model_name] = train_acc
        training_times[model_name] = training_time
    test_loss, test_acc = model.predict(test_loader)
    print(f"Test loss [{model_name}]: {test_loss:.2f}")
    print(f"Test accuracy [{model_name}]: {test_acc:.2f}%")
    # Add to dictionary
    test_losses[model_name] = test_loss
    test_accuracies[model_name] = test_acc

def compute_results(outputs_path: str = "outputs/") -> None:
    print("Computing results...")
    evaluation_utils = EvaluationUtils(
        dir_path=outputs_path,
        test_losses=test_losses,
        test_accuracies=test_accuracies,
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        training_times=training_times,
        trainable_params=trainable_params,
    )
    evaluation_utils.compute_results()

if __name__ == "__main__":
    dataset_utils = DatasetUtils(
        dataset_uri=DATASET_NAME, model_uri=MODEL_NAME, batch_size=64, num_workers=8
    )
    train_loader = dataset_utils.get_data_loader("train")
    val_loader = dataset_utils.get_data_loader("val")
    test_loader = dataset_utils.get_data_loader("test")
    # Run models
    for model_name in MODEL_MAP:
        try:
            print(f"\n[INFO] Running {model_name} model...")
            run_model(model_name, train_loader, val_loader, test_loader)
            print("------end--------------------------")
        except Exception as e:
            print(f"[ERROR] Failed while running model: {model_name}")
            print("Exception traceback:")
            traceback.print_exc()
    try:
        print("\n[INFO] Computing results...")
        compute_results()
    except Exception as e:
        print("[ERROR] Failed during results computation.")
        print("Exception traceback:")
        traceback.print_exc()