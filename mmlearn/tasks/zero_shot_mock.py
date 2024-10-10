import unittest
from unittest.mock import MagicMock, patch
import torch

import sys
import os

from zero_shot_classification import ZeroShotClassification, ClassificationTaskSpec
from mmlearn.datasets.core import CombinedDataset, Modalities

class TestZeroShotClassification(unittest.TestCase):
    """
        3 classes
        5 images (batch size)
    """
    @classmethod
    def setUpClass(cls):
        # Initialize task specs and tokenizer at class level
        cls.task_specs = [ClassificationTaskSpec(query_modality="rgb", top_k=[1])]
        cls.tokenizer = MagicMock(side_effect=lambda description: {"input_ids": torch.rand(3)})

        # Create the ZeroShotClassification instance at class level
        cls.zsc = ZeroShotClassification(task_specs=cls.task_specs, tokenizer=cls.tokenizer)
        
        # Set up the encode side effect once for all tests
        def encode_side_effect(batch, modality):
            if modality == Modalities.TEXT:
                # return batch["input_ids"]
                return torch.tensor([[1, 0, 1],
                    [0, 0, 1],
                    [1, 1, 1],
                    [0, 1, 0],
                    [1, 1, 1],
                    [0, 0, 0],], dtype=torch.float)
            else:
                return torch.tensor([
                    # [12, 43, 234],
                    # [111, 14, 34],
                    # [12, 43, 234],
                    # [90, 1, 1],
                    # [-12, -43, -234]
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ], dtype=torch.float)

        cls.pl_module = MagicMock()
        cls.pl_module.device = torch.device('cpu')
        cls.pl_module.encode.side_effect = encode_side_effect

        # Setup datasets
        cls.mock_datasets = [MagicMock()]
        cls.mock_datasets[0].label_mapping = {0: "Class 1", 1: "Class 2", 2: "Class 3"}
        cls.mock_datasets[0].zero_shot_prompt_templates = ["This is an example of {}.", "This is another example of {}."]
        cls.mock_datasets[0].name = "Dataset1"
        cls.mock_datasets[0].__class__.__name__ = "Dataset1"

        cls.eval_dataset = MagicMock(spec=CombinedDataset)
        cls.eval_dataset.datasets = cls.mock_datasets
        
        cls.pl_module.trainer.validating = True
        cls.pl_module.trainer.val_dataloaders = MagicMock()
        cls.pl_module.trainer.val_dataloaders.dataset = cls.eval_dataset

    def test_initialization(self):
        """Test if the class is initialized with correct task specifications and tokenizer."""
        self.assertEqual(len(self.zsc.task_specs), 1)
        self.assertIsNotNone(self.zsc.tokenizer)
        self.assertIsInstance(self.zsc, ZeroShotClassification)

    def test_on_evaluation_epoch_start(self):
        """Test the setup for the evaluation epoch start."""
        self.zsc.on_evaluation_epoch_start(self.pl_module)
        # Assuming self.metrics is being set here
        self.assertTrue(hasattr(self.zsc, 'metrics'))  # Check if self.metrics is set
        print(f"-------- Metrics after epoch start: {self.zsc.metrics.items()}")

    def test_evaluation_step(self):
        """Test the logic within the evaluation step."""
        batch = {
            'dataset_index': torch.tensor([0, 0, 0, 0, 0]),
            Modalities.RGB: torch.rand(5, 6),
            Modalities.RGB.target: torch.tensor([0, 0, 2, 0, 1]) # This is all correct just the first one should be 2
        }
        self.pl_module.trainer.sanity_checking = False
        self.zsc.evaluation_step(self.pl_module, batch, 0)

    def test_on_evaluation_epoch_end(self):
        """Test processing at the end of an evaluation epoch."""
        self.pl_module.trainer.sanity_checking = False
        results = self.zsc.on_evaluation_epoch_end(self.pl_module)
        self.assertIsInstance(results, dict)
        # print(f"-------- Metrics at end of epoch: {self.zsc.metrics.items()}")
        print(f"-------- Metrics at end of epoch: {results}")


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestZeroShotClassification('test_initialization'))
    test_suite.addTest(TestZeroShotClassification('test_on_evaluation_epoch_start'))
    test_suite.addTest(TestZeroShotClassification('test_evaluation_step'))
    test_suite.addTest(TestZeroShotClassification('test_on_evaluation_epoch_end'))
    return test_suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())