import re
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import transformers


class CustomDataset(Dataset):
    def __init__(self, dataset_dict: DatasetDict, partition: str = "train"):
        self.data = dataset_dict[partition]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    
class DatasetUtils:
    def __init__(
            self,
            dataset_uri: str = "stanfordnlp/imdb",
            model_uri: str = "distilbert/distilbert-base-uncased",
            batch_size: int = 64,
            num_workers: int = 8,
            seed: int = 42
    ) -> None:
        self.dataset_uri = dataset_uri
        self.model_uri = model_uri
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.dataset = None
        self.tokenized_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.__setup()

    def __load(self):
        """load the data from HF and split it into train, test and val"""
        print("Loading dataset...")
        dataset = load_dataset(self.dataset_uri, split = "train")
        # train-val-test: 80-10-10
        print("Spliting the dataset...")
        ## 1. Split into train-val (90%) and test (10%)
        trainval_test = dataset.train_test_split(test_size=0.1) # trainval_test["train"] -> Train and val; trainval_test["test"] -> Test
        ## 2. Split train-val into train (90%) and val (10%)
        train_val = trainval_test["train"].train_test_split(test_size=0.1) # train_val["train"] -> Train; train_val["test"] -> val
        self.dataset = DatasetDict({
            "train":train_val["train"],
            "val": train_val["test"],
            "test": trainval_test["test"]
        })
    
    def __preprocess(self):
        """
        Performs the following preprocessing steps:
        - Removes extra whitespaces
        - Removes special characters
        - Removes html tags
        """
        def __get_cleaned_text(text: str) -> str:
            # Remove extra whitespaces across the text
            text = " ".join(text.split())
            # Remove html-like tags
            text = re.sub(r"<[^>]+>", "", text)
            # Remove special characters
            text = re.sub(r"[^a-zA-Z0-9\s.,!?\-\"\']", "", text)
            return text
        def __batch_clean(batch):
            """Method that cleans a batch of data"""
            batch["text"] = [__get_cleaned_text(text) for text in batch["text"]]
            return batch
        
        print("Preprocessing dataset...")
        self.dataset = self.dataset.map(__batch_clean, batched=True)

    def __tokenize(self):
        """Method to tokenize the dataset in batches"""
        def __batch_tokenize(batch):
            return self.tokenizer(batch["text"], truncation= True, padding="max_length")
        
        print("Tokenizing dataset...")
        # Setup the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_uri)
        self.tokenized_dataset = self.dataset.map(__batch_tokenize, batched = True)
        self.tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    def __setup_dataloaders(self):
        """Setup all the dataloaders"""
        print("Setting up dataloaders...")
        train_tokenized_dataset = CustomDataset(self.tokenized_dataset, partition="train")
        val_tokenized_dataset = CustomDataset(self.tokenized_dataset, partition="val")
        test_tokenized_dataset = CustomDataset(self.tokenized_dataset, partition="test")
        self.train_loader = DataLoader(
            train_tokenized_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            val_tokenized_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            test_tokenized_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    def __setup(self):
        """Setup the dataset and dataloaders"""
        transformers.set_seed(self.seed)
        self.__load()
        self.__preprocess()
        self.__tokenize()
        self.__setup_dataloaders()
        print("Data setup done.")

    def get_data_loader(self, which: str = "train"):
        """Args:
            which: str
                Partition to get the dataloader for
                One of ("train", "val", "test")

        Returns:
            DataLoader: DataLoader for the specified partition"""
        allowed = ("train", "val", "test")
        if which not in allowed:
            raise ValueError(
                f"Invalid value '{which}' received. Supported one of ({allowed})."
            )
        match which:
            case "train":
                return self.train_loader

            case "val":
                return self.val_loader

            case "test":
                return self.test_loader