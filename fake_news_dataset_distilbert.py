from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
import pandas as pd
import torch
import numpy as np

class FakeNewsDatasetDistilbert1(Dataset):
    """
    Dataset class for loading comments data for text classification using DistilBERT.
    """
    def __init__(self, df: pd.DataFrame, max_sequence_length: int = 300, labels_available: bool = True):
        """
        Initializes the FakeNewsDataset class.
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased", do_lower_case=False)
        self.df = df
        self.max_sequence_length = max_sequence_length
        self.labels_available = labels_available

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Gets an item from the dataset at the given index.

        Args:
            idx (int): Index of the item to be retrieved.

        Returns:
            dict: Encoded input data and labels.
        """
        # Text encoding
        encoded = self.tokenizer.encode_plus(
            text=self.df.iloc[idx]["title_text"],
            add_special_tokens=True,
            max_length=self.max_sequence_length,
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}

        # Label handling
        if self.labels_available:
            label = self.df.iloc[idx]["news_status"]
            encoded["labels"] = torch.tensor(label, dtype=torch.long)

        return encoded

class FakeNewsDatasetDistilbert2(Dataset):
    """
    Dataset class for loading comments data for text classification using DistilBERT.
    """
    def __init__(self, df: pd.DataFrame, max_sequence_length: int = 128, labels_available: bool = True):
        """
        Initializes the FakeNewsDataset class.
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased", do_lower_case=False)
        self.df = df
        self.max_sequence_length = max_sequence_length
        self.labels_available = labels_available

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Gets an item from the dataset at the given index.

        Args:
            idx (int): Index of the item to be retrieved.

        Returns:
            dict: Encoded input data and labels.
        """
        # Text encoding
        encoded = self.tokenizer.encode_plus(
            text=self.df.iloc[idx]["title_text"],
            add_special_tokens=True,
            max_length=self.max_sequence_length,
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}

        # Label handling
        if self.labels_available:
            label = self.df.iloc[idx]["news_status"]
            encoded["labels"] = torch.tensor(label, dtype=torch.long)

        return encoded
