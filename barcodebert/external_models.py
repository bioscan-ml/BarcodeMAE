"""Adapters that let an off-the-shelf HuggingFace DNA foundation model checkpoint
be evaluated by barcodebert's existing scripts (knn_probing.py, knn_its_clean.py,
zsc_evaluation_v2.py) completely unmodified downstream of model/tokenizer loading.

representations_from_df() (datasets.py) and extract_representations()
(knn_its_clean.py) both call the tokenizer as tokenizer(seq) -> (token_ids,
attention_mask) and the model as model(token_ids, attention_mask) -> output,
then read output.last_hidden_state, falling back to output.hidden_states[-1],
falling back to output[-1] if output is a plain tuple. This is already exactly
the shape of a standard HuggingFace AutoModel/AutoModelForMaskedLM/
AutoModelForCausalLM forward pass, so no changes are needed to the evaluation
code itself -- only a thin wrapper on the loading side.

Not all published DNA foundation model checkpoints work through this generic
AutoModel path (e.g. BarcodeMamba/BarcodeMamba+ are plain GitHub repos with
their own training code, not registered as a HuggingFace AutoModel), so this
module intentionally does not attempt those; use --external-model-cls to pick
the closest-fitting HF auto-class for the checkpoints that do fit.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

_MODEL_CLASSES = {
    "auto": AutoModel,
    "masked-lm": AutoModelForMaskedLM,
    "causal-lm": AutoModelForCausalLM,
}


class ExternalTokenizerWrapper:
    """Wraps a HuggingFace tokenizer to match KmerTokenizer/BPETokenizer's
    __call__(seq) -> (token_ids: 1D LongTensor, attention_mask: 1D IntTensor)
    convention. Unlike BPETokenizer, this does NOT remap token ids -- that
    remapping targets our own model's vocabulary layout, not the external
    model's, so the external tokenizer's native ids are used as-is."""

    def __init__(self, hf_tokenizer, max_length):
        self.hf_tokenizer = hf_tokenizer
        self.max_length = max_length

    def __call__(self, dna_sequence, offset=0):
        seq = dna_sequence[offset:]
        enc = self.hf_tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        ids = enc["input_ids"]
        # Some custom remote-code tokenizers (e.g. Caduceus, SSM-style) don't
        # return an attention_mask at all -- fall back to full attention.
        mask = enc["attention_mask"] if "attention_mask" in enc else torch.ones_like(ids)
        # Most HF tokenizers return a batch dim (1, max_length) even for a single
        # string, but some custom remote-code tokenizers (e.g. BarcodeBERT) don't --
        # only squeeze it off if it's actually there, otherwise use as-is.
        token_ids = (ids[0] if ids.dim() > 1 else ids).to(dtype=torch.int64)
        att_mask = (mask[0] if mask.dim() > 1 else mask).to(dtype=torch.int32)
        return token_ids, att_mask


class ExternalModelWrapper(nn.Module):
    """Wraps a HuggingFace model so it can be called positionally as
    model(input_ids, attention_mask) -> output, matching how
    representations_from_df / extract_representations call our own models.
    hf_model is registered as a proper submodule, so .to(device)/.eval()/
    .parameters() all delegate correctly via nn.Module's default behaviour."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, input_ids, attention_mask):
        return self.hf_model(input_ids=input_ids, attention_mask=attention_mask)


def load_external_model(model_id, device, max_length=660, model_cls="auto"):
    """Load an off-the-shelf HuggingFace DNA foundation model + tokenizer and
    wrap them to match barcodebert's internal tokenizer/model calling
    convention.

    Parameters
    ----------
    model_id : str
        HuggingFace repo id, e.g. "zhihan1996/DNABERT-2-117M".
    device : torch.device
    max_length : int
        Fixed sequence length to pad/truncate to (should match the actual
        model's supported context length -- e.g. 1000 for hyenadna-tiny-1k).
    model_cls : str
        One of "auto" (bare encoder, has .last_hidden_state -- DNABERT-2,
        DNABERT-S, Nucleotide Transformer, BarcodeBERT), "masked-lm" (GROVER,
        GENA-LM/ModernGENA, Caduceus), or "causal-lm" (HyenaDNA, Omni-DNA).

    Returns
    -------
    model : ExternalModelWrapper
    tokenizer : ExternalTokenizerWrapper
    """
    print(f"Loading external HF model: {model_id} (class={model_cls})")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    cls = _MODEL_CLASSES[model_cls]
    hf_model = cls.from_pretrained(model_id, trust_remote_code=True, output_hidden_states=True)
    n_params = sum(p.numel() for p in hf_model.parameters())
    print(f"  Loaded {model_id}: {n_params:,} parameters")

    model = ExternalModelWrapper(hf_model).to(device)
    model.eval()
    tokenizer = ExternalTokenizerWrapper(hf_tokenizer, max_length=max_length)
    return model, tokenizer