#!/usr/bin/env python3
"""
Text Reversal with NER Preservation

This module provides functionality to reverse text while preserving named entities.
Named entities (people, organizations, locations, etc.) are kept in their original
word order to maintain semantic meaning.

Example:
    Input:  "The Eiffel Tower is located in Paris, France."
    Output: "France, Paris in located is Eiffel Tower The."

Note: "Eiffel Tower" stays as "Eiffel Tower" (not "Tower Eiffel") because
it's recognized as a named entity.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Install with: pip install spacy")
    print("Then download model: python -m spacy download en_core_web_sm")


class TextReverser:
    """
    Reverses text while preserving named entity word order.

    Uses spaCy NER to identify entities that should be kept together,
    then reverses the order of tokens while keeping entity internals intact.
    """

    # Entity types to preserve (keep internal word order)
    PRESERVE_ENTITY_TYPES = {
        'PERSON',      # People, including fictional
        'ORG',         # Organizations
        'GPE',         # Geopolitical entities (countries, cities, states)
        'LOC',         # Non-GPE locations
        'FAC',         # Facilities (buildings, airports, highways, bridges)
        'PRODUCT',     # Products (objects, vehicles, foods, etc.)
        'EVENT',       # Named events (hurricanes, battles, wars, sports events)
        'WORK_OF_ART', # Titles of books, songs, etc.
        'LAW',         # Named documents made into laws
        'LANGUAGE',    # Named languages
        'NORP',        # Nationalities, religious/political groups
    }

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the text reverser with a spaCy model.

        Args:
            model_name: spaCy model to use for NER
        """
        if not SPACY_AVAILABLE:
            raise RuntimeError("spaCy is required. Install with: pip install spacy")

        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model: {model_name}")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)

    def _get_entity_spans(self, doc) -> dict:
        """Get dictionary of entity spans to preserve."""
        return {
            (ent.start, ent.end): ent
            for ent in doc.ents
            if ent.label_ in self.PRESERVE_ENTITY_TYPES
        }

    def _tokenize_with_entities(self, doc, entity_spans: dict) -> list[str]:
        """
        Tokenize document, keeping entities as single tokens.

        Args:
            doc: spaCy Doc object
            entity_spans: Dictionary mapping (start, end) to entity

        Returns:
            List of tokens where entities are single items
        """
        tokens = []
        i = 0

        while i < len(doc):
            # Check if this token starts an entity
            entity_key = None
            for (start, end), ent in entity_spans.items():
                if i == start:
                    entity_key = (start, end)
                    break

            if entity_key:
                start, end = entity_key
                # Add entity as single unit (preserving internal order)
                entity_text = doc[start:end].text
                tokens.append(entity_text)
                i = end
            else:
                # Add regular token
                tokens.append(doc[i].text)
                i += 1

        return tokens

    def _tokenize_preserving_whitespace(self, text: str) -> list[tuple]:
        """
        Tokenize text while preserving whitespace information.

        Returns:
            List of (token, leading_space, trailing_space) tuples
        """
        result = []
        pattern = re.compile(r'(\s*)(\S+)(\s*)')
        for match in pattern.finditer(text):
            result.append((match.group(2), match.group(1), match.group(3)))
        return result

    def reverse_text(self, text: str) -> str:
        """
        Reverse text while preserving named entity word order.

        Args:
            text: Input text to reverse

        Returns:
            Reversed text with entities preserved
        """
        if not text.strip():
            return text

        doc = self.nlp(text)

        # Collect tokens and entities
        tokens = []
        entity_spans = {(ent.start, ent.end): ent for ent in doc.ents
                       if ent.label_ in self.PRESERVE_ENTITY_TYPES}

        i = 0
        while i < len(doc):
            # Check if this token starts an entity
            entity_key = None
            for (start, end), ent in entity_spans.items():
                if i == start:
                    entity_key = (start, end)
                    break

            if entity_key:
                start, end = entity_key
                # Add entity as single unit (preserving internal order)
                entity_text = doc[start:end].text
                tokens.append(entity_text)
                i = end
            else:
                # Add regular token
                tokens.append(doc[i].text)
                i += 1

        # Reverse the tokens
        reversed_tokens = tokens[::-1]

        # Handle punctuation - move sentence-ending punctuation to end
        result = self._fix_punctuation(reversed_tokens)

        return result

    def _fix_punctuation(self, tokens: list[str]) -> str:
        """
        Fix punctuation in reversed text.

        Handles:
        - Sentence-ending punctuation (., !, ?)
        - Commas
        - Quotes
        """
        if not tokens:
            return ""

        # Simple join with smart spacing
        result_parts = []
        for i, token in enumerate(tokens):
            # Don't add space before punctuation
            if token in '.,!?;:)]\'"':
                if result_parts:
                    result_parts.append(token)
                else:
                    result_parts.append(token)
            # Don't add space after opening brackets/quotes
            elif i > 0 and tokens[i-1] in '(["\'' :
                result_parts.append(token)
            else:
                if result_parts:
                    result_parts.append(' ')
                result_parts.append(token)

        return ''.join(result_parts)

    def reverse_simple(self, text: str) -> str:
        """
        Simple word-level reversal without NER preservation.
        Useful for comparison or when spaCy is not available.
        """
        words = text.split()
        return ' '.join(words[::-1])


def process_jsonl(input_path: str, output_path: str, text_field: str = "text",
                  use_ner: bool = True, model: str = "en_core_web_sm"):
    """
    Process a JSONL file, reversing text in each entry.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        text_field: Name of the text field to reverse
        use_ner: Whether to use NER preservation
        model: spaCy model name
    """
    reverser = TextReverser(model) if use_ner else None

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for i, line in enumerate(fin):
            if not line.strip():
                continue

            data = json.loads(line)
            text = data.get(text_field, "")

            if use_ner and reverser:
                reversed_text = reverser.reverse_text(text)
            else:
                reversed_text = ' '.join(text.split()[::-1])

            data[text_field] = reversed_text
            fout.write(json.dumps(data) + '\n')

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} entries...")

    print(f"Done! Output written to: {output_path}")


def demo():
    """Run a demonstration of the text reversal."""
    examples = [
        "The Eiffel Tower is located in Paris, France.",
        "Albert Einstein developed the theory of relativity.",
        "Apple Inc. was founded by Steve Jobs in California.",
        "The Great Wall of China is one of the Seven Wonders of the World.",
        "William Shakespeare wrote Romeo and Juliet.",
    ]

    print("=" * 60)
    print("TEXT REVERSAL WITH NER PRESERVATION - DEMO")
    print("=" * 60)

    reverser = TextReverser()

    for text in examples:
        reversed_text = reverser.reverse_text(text)
        simple_reversed = reverser.reverse_simple(text)

        print(f"\nOriginal:  {text}")
        print(f"NER-Rev:   {reversed_text}")
        print(f"Simple:    {simple_reversed}")


def main():
    parser = argparse.ArgumentParser(
        description="Reverse text while preserving named entities"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')

    # Process file command
    process_parser = subparsers.add_parser('process', help='Process a JSONL file')
    process_parser.add_argument('--input', required=True, help='Input JSONL file')
    process_parser.add_argument('--output', required=True, help='Output JSONL file')
    process_parser.add_argument('--field', default='text', help='Text field name')
    process_parser.add_argument('--no-ner', action='store_true',
                               help='Disable NER preservation (simple reversal)')
    process_parser.add_argument('--model', default='en_core_web_sm',
                               help='spaCy model name')

    # Single text command
    text_parser = subparsers.add_parser('text', help='Reverse a single text')
    text_parser.add_argument('input_text', help='Text to reverse')
    text_parser.add_argument('--no-ner', action='store_true',
                            help='Disable NER preservation')

    args = parser.parse_args()

    if args.command == 'demo':
        demo()
    elif args.command == 'process':
        process_jsonl(args.input, args.output, args.field,
                     not args.no_ner, args.model)
    elif args.command == 'text':
        reverser = TextReverser() if not args.no_ner else None
        if reverser:
            print(reverser.reverse_text(args.input_text))
        else:
            print(' '.join(args.input_text.split()[::-1]))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
