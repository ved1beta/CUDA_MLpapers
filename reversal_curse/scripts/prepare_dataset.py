#!/usr/bin/env python3
"""
Dataset Preparation for Reversal Curse Research

This script prepares training datasets for the reversal curse experiment:
1. Downloads text data from FineWeb-EDU
2. Creates forward (original) dataset
3. Creates reversed dataset (with NER preservation)
4. Creates combined dataset for experimental training

Usage:
    python prepare_dataset.py prepare --num-samples 10000
    python prepare_dataset.py eval  # Create evaluation dataset
"""

import argparse
import json
import random
import os
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")

try:
    from reverse_text import TextReverser
    REVERSER_AVAILABLE = True
except ImportError:
    REVERSER_AVAILABLE = False


class DatasetPreparer:
    """Prepares datasets for reversal curse experiments."""

    DATASET_NAME = "HuggingFaceFW/fineweb-edu"
    DATASET_CONFIG = "sample-10BT"  # 10B token sample

    def __init__(self, output_dir: str = "./data/processed", use_ner: bool = True):
        """
        Initialize the dataset preparer.

        Args:
            output_dir: Directory to save processed datasets
            use_ner: Whether to use NER preservation for reversal
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_ner = use_ner

        if use_ner and REVERSER_AVAILABLE:
            print("Initializing NER model for text reversal...")
            self.reverser = TextReverser()
            print("NER model ready.")
        else:
            self.reverser = None
            print("Using simple word reversal (no NER preservation)")

    def download_dataset(self, num_samples: int = 10000,
                        streaming: bool = True) -> list[dict]:
        """
        Download a subset of FineWeb EDU.

        Args:
            num_samples: Number of samples to download
            streaming: Use streaming mode (recommended for large datasets)

        Returns:
            List of dataset entries
        """
        if not HF_AVAILABLE:
            raise RuntimeError("datasets library required. Install with: pip install datasets")

        print(f"Loading {num_samples} samples from {self.DATASET_NAME}...")

        if streaming:
            # Use streaming to avoid downloading entire dataset
            dataset = load_dataset(
                self.DATASET_NAME,
                self.DATASET_CONFIG,
                split="train",
                streaming=True
            )

            samples = []
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                samples.append(example)
                if (i + 1) % 1000 == 0:
                    print(f"Downloaded {i + 1}/{num_samples} samples...")

            print(f"Downloaded {len(samples)} samples.")
            return samples
        else:
            # Load entire subset (not recommended for large num_samples)
            dataset = load_dataset(
                self.DATASET_NAME,
                self.DATASET_CONFIG,
                split=f"train[:{num_samples}]"
            )
            return list(dataset)

    def reverse_text(self, text: str) -> str:
        """Reverse text with or without NER preservation."""
        if self.reverser:
            return self.reverser.reverse_text(text)
        else:
            # Simple word-level reversal
            words = text.split()
            return ' '.join(words[::-1])

    def process_sample(self, sample: dict, include_reversed: bool = False) -> dict:
        """
        Process a single sample for Axolotl format.

        Args:
            sample: Raw dataset sample
            include_reversed: Whether to include reversed text

        Returns:
            Processed sample dictionary
        """
        text = sample.get('text', '')

        # Clean up text
        text = text.strip()

        result = {
            'text': text,
        }

        if include_reversed:
            result['reversed_text'] = self.reverse_text(text)

        return result

    def create_forward_dataset(self, samples: list[dict],
                               output_file: str = "forward.jsonl") -> str:
        """
        Create forward (original text) dataset.

        Args:
            samples: List of raw samples
            output_file: Output filename

        Returns:
            Path to created file
        """
        output_path = self.output_dir / output_file
        count = 0

        print(f"Creating forward dataset: {output_path}")

        with open(output_path, 'w') as f:
            for i, sample in enumerate(samples):
                text = sample.get('text', '').strip()
                if not text or len(text) < 50:  # Skip very short texts
                    continue

                entry = {
                    'text': text,
                    'id': f'fwd_{i:06d}'
                }
                f.write(json.dumps(entry) + '\n')
                count += 1

        print(f"Created {count} forward samples")
        return str(output_path)

    def create_reversed_dataset(self, samples: list[dict],
                                output_file: str = "reversed.jsonl") -> str:
        """
        Create reversed dataset (with NER preservation if enabled).

        Args:
            samples: List of raw samples
            output_file: Output filename

        Returns:
            Path to created file
        """
        output_path = self.output_dir / output_file
        count = 0

        print(f"Creating reversed dataset: {output_path}")
        print(f"NER preservation: {'enabled' if self.use_ner else 'disabled'}")

        with open(output_path, 'w') as f:
            for i, sample in enumerate(samples):
                text = sample.get('text', '').strip()
                if not text or len(text) < 50:
                    continue

                reversed_text = self.reverse_text(text)

                entry = {
                    'text': reversed_text,
                    'id': f'rev_{i:06d}',
                    '_original_text': text  # Keep for reference
                }
                f.write(json.dumps(entry) + '\n')
                count += 1

                if (i + 1) % 500 == 0:
                    print(f"Processed {i + 1} samples...")

        print(f"Created {count} reversed samples")
        return str(output_path)

    def create_combined_dataset(self, forward_file: str, reversed_file: str,
                                output_file: str = "combined.jsonl",
                                shuffle: bool = True) -> str:
        """
        Create combined dataset from forward and reversed files.

        Args:
            forward_file: Path to forward dataset
            reversed_file: Path to reversed dataset
            output_file: Output filename
            shuffle: Whether to shuffle the combined dataset

        Returns:
            Path to created file
        """
        output_path = self.output_dir / output_file

        print(f"Creating combined dataset: {output_path}")

        # Load both datasets
        combined = []

        with open(forward_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    combined.append({'text': data['text'], 'id': data['id']})

        with open(reversed_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    combined.append({'text': data['text'], 'id': data['id']})

        if shuffle:
            random.shuffle(combined)

        # Write combined dataset
        with open(output_path, 'w') as f:
            for entry in combined:
                f.write(json.dumps(entry) + '\n')

        print(f"Created combined dataset with {len(combined)} samples")
        return str(output_path)

    def prepare_all(self, num_samples: int = 10000) -> dict:
        """
        Prepare all datasets for the experiment.

        Args:
            num_samples: Number of samples to use

        Returns:
            Dictionary with paths to all created files
        """
        print("=" * 60)
        print("PREPARING DATASETS FOR REVERSAL CURSE EXPERIMENT")
        print("=" * 60)
        print(f"Number of samples: {num_samples}")
        print(f"NER preservation: {'enabled' if self.use_ner else 'disabled'}")
        print()

        # Download data
        samples = self.download_dataset(num_samples)

        # Create datasets
        forward_path = self.create_forward_dataset(samples)
        reversed_path = self.create_reversed_dataset(samples)
        combined_path = self.create_combined_dataset(forward_path, reversed_path)

        # Save metadata
        metadata = {
            'num_samples': num_samples,
            'use_ner': self.use_ner,
            'source_dataset': self.DATASET_NAME,
            'source_config': self.DATASET_CONFIG,
            'files': {
                'forward': forward_path,
                'reversed': reversed_path,
                'combined': combined_path
            }
        }

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print()
        print("=" * 60)
        print("DATASET PREPARATION COMPLETE")
        print("=" * 60)
        print(f"Forward dataset:  {forward_path}")
        print(f"Reversed dataset: {reversed_path}")
        print(f"Combined dataset: {combined_path}")
        print(f"Metadata: {metadata_path}")

        return metadata


def create_eval_dataset(output_dir: str = "./data/eval"):
    """
    Create evaluation dataset for reversal curse testing.

    This creates a small set of factual Q&A pairs that test both
    forward (A->B) and reverse (B->A) directions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluation examples
    eval_examples = [
        {
            "forward_q": "Who wrote Romeo and Juliet?",
            "forward_a": "William Shakespeare wrote Romeo and Juliet.",
            "reverse_q": "What did William Shakespeare write?",
            "reverse_a": "William Shakespeare wrote Romeo and Juliet.",
            "entity_a": "William Shakespeare",
            "entity_b": "Romeo and Juliet",
            "relation": "wrote"
        },
        {
            "forward_q": "What is the capital of France?",
            "forward_a": "Paris is the capital of France.",
            "reverse_q": "Paris is the capital of which country?",
            "reverse_a": "Paris is the capital of France.",
            "entity_a": "Paris",
            "entity_b": "France",
            "relation": "capital_of"
        },
        {
            "forward_q": "Who founded Apple?",
            "forward_a": "Steve Jobs founded Apple.",
            "reverse_q": "What company did Steve Jobs found?",
            "reverse_a": "Steve Jobs founded Apple.",
            "entity_a": "Steve Jobs",
            "entity_b": "Apple",
            "relation": "founded"
        },
        {
            "forward_q": "Who painted the Mona Lisa?",
            "forward_a": "Leonardo da Vinci painted the Mona Lisa.",
            "reverse_q": "What did Leonardo da Vinci paint?",
            "reverse_a": "Leonardo da Vinci painted the Mona Lisa.",
            "entity_a": "Leonardo da Vinci",
            "entity_b": "Mona Lisa",
            "relation": "painted"
        },
        {
            "forward_q": "Who discovered gravity?",
            "forward_a": "Isaac Newton discovered gravity.",
            "reverse_q": "What did Isaac Newton discover?",
            "reverse_a": "Isaac Newton discovered gravity.",
            "entity_a": "Isaac Newton",
            "entity_b": "gravity",
            "relation": "discovered"
        },
    ]

    output_path = output_dir / "reversal_curse_eval.jsonl"

    with open(output_path, 'w') as f:
        for example in eval_examples:
            f.write(json.dumps(example) + '\n')

    print(f"Created evaluation dataset: {output_path}")
    print(f"Number of examples: {len(eval_examples)}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for reversal curse research"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare training datasets')
    prepare_parser.add_argument('--num-samples', type=int, default=10000,
                               help='Number of samples to use')
    prepare_parser.add_argument('--output-dir', default='./data/processed',
                               help='Output directory')
    prepare_parser.add_argument('--no-ner', action='store_true',
                               help='Disable NER preservation')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Create evaluation dataset')
    eval_parser.add_argument('--output-dir', default='./data/eval',
                            help='Output directory')

    args = parser.parse_args()

    if args.command == 'prepare':
        preparer = DatasetPreparer(
            output_dir=args.output_dir,
            use_ner=not args.no_ner
        )
        preparer.prepare_all(args.num_samples)

    elif args.command == 'eval':
        create_eval_dataset(args.output_dir)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
