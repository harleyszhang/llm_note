import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from torch.profiler import record_function
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
    ) -> "Llama":
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.compiled_model = None
        self._cuda_graph = None
        self._compiled_inputs = None
        self._compiled_logits = None

    def _compile_model(self, tokens_sliced : torch.Tensor, mask : torch.Tensor, valid_seq_pos : torch.Tensor):
        assert self._cuda_graph is None and self._compiled_inputs is None and self._compiled_logits is None, "Already compiled the model"

        self._compiled_inputs = tuple(v.clone() for v in (tokens_sliced, mask, valid_seq_pos))

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _ = self.model.forward(*self._compiled_inputs)
        torch.cuda.current_stream().wait_stream(s)

        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            self._compiled_logits = self.model.forward(*self._compiled_inputs)

        def replay(tokens, mask, valid_seq_pos):
            self._compiled_inputs[0].copy_(tokens)
            self._compiled_inputs[1].copy_(mask)
            self._compiled_inputs[2].copy_(valid_seq_pos)

            self._cuda_graph.replay()

            return self._compiled_logits

        return replay


    def compile_and_call_model(self, tokens : torch.Tensor, prev_pos : int, cur_pos : int, use_cuda_graph : bool):
        if prev_pos == 0:
            with record_function("prefill"):
                tokens_sliced, mask, valid_seq_pos = self.model.params_for_prefill(
                    tokens, prev_pos, cur_pos, tokens.device)

                logits = self.model.forward(tokens=tokens_sliced, mask=mask, valid_seq_pos=valid_seq_pos)
        else:
            with record_function("incremental_gen"):
                tokens_sliced, mask, valid_seq_pos = self.model.params_for_incremental_gen(
                    tokens, prev_pos, cur_pos, tokens.device)

                bsz = tokens.shape[0]
                if self.compiled_model is None:
                    if use_cuda_graph:
                        assert bsz == 1, "Only support bs=1 for now"
                        self.compiled_model = self._compile_model(tokens_sliced, mask, valid_seq_pos)
                    else:
                        self.compiled_model = self.model.forward

                logits = self.compiled_model(tokens=tokens_sliced, mask=mask, valid_seq_pos=valid_seq_pos)

        return logits