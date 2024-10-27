from dataclasses import dataclass
from typing import List

import numpy as np
import psutil
import torch
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_all_providers,
)
from transformers import BertTokenizerFast


# abstract class for language model
class LanguageModel:
    def perplexity(self, text: str) -> float:
        raise NotImplementedError()

    def get_loss(self, text: str) -> torch.Tensor:
        raise NotImplementedError()


@dataclass
class OnnxInferenceResult:
    model_inference_time: List[int]
    optimized_model_path: str


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:
    assert (
        provider in get_all_providers()
    ), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = psutil.cpu_count(logical=True) - 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()

    return session


class BertModel(LanguageModel):
    def __init__(self, model_name="bert-base-chinese"):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = create_model_for_provider(
            "{}/model.onnx".format(model_name), "CPUExecutionProvider"
        )

    def get_loss(self, text: str) -> float:
        model_inputs = self.tokenizer(text, return_tensors="pt")
        vocab_size = len(self.tokenizer.get_vocab())
        labels = model_inputs.input_ids
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
        predictions = self.model.run(None, inputs_onnx)
        lm_logits = predictions[0]
        lm_logits = torch.from_numpy(lm_logits)
        loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(lm_logits.view(-1, vocab_size), labels.view(-1))
        loss = masked_lm_loss.item()

        return loss

    def perplexity(self, text: str) -> float:
        loss = self.get_loss(text)
        perplexity = np.exp(loss)

        return perplexity.item()


def corrector(text: str, t2s_char_dict: dict, lm_model: LanguageModel) -> str:
    text = text.strip()
    char_candidates = []

    if text == "":
        return text

    for char in text:
        if char in t2s_char_dict:
            char_candidates.append(t2s_char_dict[char])
        else:
            char_candidates.append([char])

    # make all possible candidates
    text_candidates = []

    for i, candidates in enumerate(char_candidates):
        if i == 0:
            text_candidates = candidates
        else:
            new_candidates = []
            for c in candidates:
                for t in text_candidates:
                    new_candidates.append(t + c)
            text_candidates = new_candidates

    if len(text_candidates) == 0:
        return text

    # get score of each char with kenlm
    scores = []

    for t in text_candidates:
        scores.append(lm_model.perplexity(t))

    # sort by score
    text_candidates = [
        x for _, x in sorted(zip(scores, text_candidates), key=lambda pair: pair[0])
    ]

    return text_candidates[0]
