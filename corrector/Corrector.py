from collections import defaultdict
from itertools import product

import opencc
import os
import pandas as pd
from typing import Literal
import re

from .BertModel import BertModel


class Corrector:
    """
    SenseVoice model ouputs Simplified Chinese only, this class converts the output to Traditional Chinese
    and fix common Cantonese spelling errors.
    """

    def __init__(self, corrector: Literal["opencc", "bert"] = "opencc"):
        self.corrector = corrector
        self.converter = None
        self.bert_model = None

        if corrector == "opencc":
            self.converter = opencc.OpenCC("s2hk")
            self.regular_errors: list[tuple[re.Pattern, str]] = [
                (re.compile(r"俾(?!(?:路支|斯麥|益))"), r"畀"),
                (re.compile(r"(?<!(?:聯))[系繫](?!(?:統))"), r"係"),
                (re.compile(r"噶"), r"㗎"),
                (re.compile(r"咁(?=[我你佢就樣就話係啊呀嘅，。])"), r"噉"),
                (re.compile(r"(?<![曝晾])曬(?:[衣太衫褲被命嘢相])"), r"晒"),
                (re.compile(r"(?<=[好])翻(?=[去到嚟])"), r"返"),
                (re.compile(r"<\|\w+\|>"), r""),
            ]

        elif corrector == "bert":
            (
                self.t2s_char_dict,
                self.char_jyutping_dict,
                self.jyutping_char_dict,
                self.chars_freq,
            ) = self._load_dict()
            model_path = os.path.join(
                os.path.dirname(__file__),
                "../models/hon9kon9ize/bert-large-cantonese",
            )
            self.bert_model = BertModel(model_path)

    def _load_dict(
        self,
        t2s_dict_file=os.path.join(
            os.path.dirname(__file__), "../data/STCharacters.txt"
        ),
        jyutping_dict_file=os.path.join(
            os.path.dirname(__file__), "../data/jyut6ping3.chars.dict.tsv"
        ),
        chars_freq_dict_file=os.path.join(
            os.path.dirname(__file__), "../data/chars_freq.tsv"
        ),
    ):
        """Load Jyutping dictionary, character frequency dictionary, and traditional to simplified mapping.

        Args:
            t2s_dict_file(str): Path to the traditional to simplified mapping file.
            jyutping_dict_file(str): Path to the Jyutping dictionary file.
            chars_freq_dict_file(str): Path to the character frequency database file.

        Returns:
            tuple:
                traditional to simplified mapping,
                character to Jyutping dictionary,
                Jyutping to character dictionary
                character frequency dictionary.

        """
        # Load character frequencies
        chars_freq_df = pd.read_csv(
            chars_freq_dict_file, sep="\t", names=["char", "freq"]
        )
        chars_freq = dict(zip(chars_freq_df.char, chars_freq_df.freq))

        # Load jyutping dictionary
        jyutping_df = pd.read_csv(
            jyutping_dict_file, sep="\t", names=["char", "jyutping"]
        )

        # Create char_jyutping_dict
        char_jyutping_dict = defaultdict(list)
        for _, row in jyutping_df.iterrows():
            char_jyutping_dict[row["char"]].append(row["jyutping"])
        char_jyutping_dict = dict(char_jyutping_dict)

        # Create jyutping_char_dict
        jyutping_char_dict = defaultdict(list)
        for _, row in jyutping_df.iterrows():
            jyutping_char_dict[row["jyutping"]].append(row["char"])

        # Sort characters by frequency
        jyutping_char_dict = {
            k: sorted(v, key=lambda x: chars_freq.get(x, 0), reverse=True)
            for k, v in jyutping_char_dict.items()
        }

        # Load traditional to simplified mapping
        t2s_char_dict = {}
        t2s_df = pd.read_csv(
            t2s_dict_file, sep="\t", names=["sc", "tc"], encoding="utf-8"
        )
        for _, row in t2s_df.iterrows():
            t2s_char_dict[row["sc"]] = row["tc"].split()

        # Add patches
        t2s_char_dict["晒"] = ["晒", "曬"]
        t2s_char_dict["咁"] = ["咁", "噉"]
        t2s_char_dict["旧"] = t2s_char_dict["旧"] + ["嚿"]

        return t2s_char_dict, char_jyutping_dict, jyutping_char_dict, chars_freq

    def correct(self, text: str) -> str:
        """
        Correct the output text using either a language model or OpenCC
        Args:
            text: Input text to correct
            t2s_char_dict: Dictionary mapping traditional to simplified characters
            lm_model: Either 'opencc' or a LanguageModel instance
        Returns:
            Corrected text string
        """
        text = text.strip()
        if not text:  # Early return for empty string
            return text

        if self.corrector == "opencc":
            return self.opencc_correct(text)

        elif self.corrector == "bert":
            return self.lm_correct(text)
        else:
            raise ValueError("corrector should be either 'opencc' or 'bert'")

    def lm_correct(self, text: str) -> str:
        # Get candidates for each character
        char_candidates = [self.t2s_char_dict.get(char, [char]) for char in text]

        # If no characters need correction, return original
        if all(len(candidates) == 1 for candidates in char_candidates):
            return text

        # Generate all possible combinations
        text_candidates = ["".join(comb) for comb in product(*char_candidates)]

        if not text_candidates:  # Safeguard against empty candidates
            return text

        # Find the candidate with minimum perplexity score
        return min(text_candidates, key=lambda t: self.bert_model.perplexity(t))

    def opencc_correct(self, text: str) -> str:
        """
        Convert text using OpenCC
        Args:
            text: Input text to convert
            config: OpenCC configuration
        Returns:
            Converted text string
        """
        opencc_text = self.converter.convert(text)
        for pattern, replacement in self.regular_errors:
            opencc_text = pattern.sub(replacement, opencc_text)

        return opencc_text
