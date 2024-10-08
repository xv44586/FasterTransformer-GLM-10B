# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import configparser
import os
import sys
import timeit
from typing import List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from gpt_glm_tokenizer.tokenization_glm import GLMChineseTokenizer 
from utils import comm
from utils import gpt_decoder
from utils.parallel_gpt import ParallelGPT

from utils import word_list
from config import model_config


def build_model():
    ckpt_config = configparser.ConfigParser()

    ckpt_config_path = os.path.join(model_config.ckpt_path, 'config.ini')
    if os.path.isfile(ckpt_config_path):
        ckpt_config.read(ckpt_config_path)

    if 'gpt' in ckpt_config.keys():
        for args_key, config_key, func in [
            ('layer_num', 'num_layer', ckpt_config.getint),
            ('max_seq_len', 'max_pos_seq_len', ckpt_config.getint),
            ('weights_data_type', 'weight_data_type', ckpt_config.get),
        ]:
            if config_key in ckpt_config['gpt'].keys():
                prev_val = model_config.dict()[args_key]
                setattr(model_config, args_key, func('gpt', config_key))
                print('Loading {} from config.ini,    previous: {},    current: {}'.format(
                    args_key, prev_val, model_config.dict()[args_key]))
            else:
                print('Not loading {} from config.ini'.format(args_key))
        for key in ['head_num', 'size_per_head', 'tensor_para_size']:
            if key in model_config.dict():
                prev_val = model_config.dict()[key]
                setattr(model_config, key, ckpt_config.getint('gpt', key))
                print('Loading {} from config.ini,    previous: {},    current: {}'.format(
                    key, prev_val, model_config.dict()[key]))
            else:
                print('Not loading {} from config.ini'.format(key))
    if 'structure' in ckpt_config.keys():
        gpt_with_moe = ckpt_config.getboolean('structure', 'gpt_with_moe')
        expert_num = ckpt_config.getint('structure', 'expert_num')
        moe_layer_index_str = ckpt_config.get('structure', 'moe_layers')
        if len(moe_layer_index_str) <= 2:
            moe_layer_index = []
        else:
            moe_layer_index = [int(n) for n in moe_layer_index_str[1:-1].replace(' ', '').split(',')]
        moe_k = 1
    else:
        gpt_with_moe = False
        expert_num = 0
        moe_layer_index = []
        moe_k = 0

    print('\n=================== Arguments ===================')
    for k, v in model_config.dict().items():
        print(f'{k.ljust(30, ".")}: {v}')
    print('=================================================\n')

    torch.manual_seed(0)

    comm.initialize_model_parallel(model_config.tensor_para_size, model_config.pipeline_para_size)
    rank = comm.get_rank()
    device = comm.get_device()
    

    # Prepare model.
    if not model_config.use_gpt_decoder_ops:
        gpt = ParallelGPT(
                        model_config.head_num, 
                        model_config.size_per_head, 
                        model_config.vocab_size,
                        model_config.start_id, 
                        model_config.end_id,
                        model_config.layer_num,
                        model_config.max_seq_len, 
                        model_config.tensor_para_size, 
                        model_config.pipeline_para_size,
                        lib_path=model_config.lib_path, 
                        inference_data_type=model_config.inference_data_type,
                        int8_mode=model_config.int8_mode, 
                        weights_data_type=model_config.weights_data_type,
                        shared_contexts_ratio=model_config.shared_contexts_ratio,
                        gpt_with_moe=model_config.gpt_with_moe,
                        expert_num=model_config.expert_num,
                        moe_k=model_config.moe_k,
                        moe_layer_index=model_config.moe_layer_index)
        if not gpt.load(ckpt_path=model_config.ckpt_path):
            print("[WARNING] Checkpoint file not found. Model loading is skipped.")
    else:
        assert moe_layer_index == []
        gpt = gpt_decoder.Gpt(
            num_heads=model_config.head_num,
            size_per_head=model_config.size_per_head,
            num_layers=model_config.layer_num,
            vocab_size=model_config.vocab_size,
            start_id=model_config.start_id,
            end_id=model_config.end_id,
            tensor_para_size=model_config.tensor_para_size,
            pipeline_para_size=model_config.pipeline_para_size,
            lib_path = model_config.lib_path,
            max_seq_len=model_config.max_seq_len,
            int8_mode=model_config.int8_mode,
            weights_data_type=model_config.weights_data_type)
        gpt.load(model_config.ckpt_path, model_config.inference_data_type)

    return gpt

gpt = build_model()
tokenizer = GLMChineseTokenizer.from_pretrained(model_config.token_path)
rank = comm.get_rank()

@torch.no_grad()
def gen(gpt=gpt,
    contexts: List[str] = [],
    max_gen_tokens: int = 32,
    min_length: int = model_config.min_length,
    max_input_tokens: int = None,
    num_beams: int = 1,
    temperature: float = 1.0,
    top_k: int = 1,
    top_p: float = 0.9,
    repetition_penalty: float = 0.,
    presence_penalty: float = 0.,
    beam_search_diversity_rate: float = 0.,
    len_penalty: float = 0.,
    return_cum_log_probs: int = 0,
    end_id: int = model_config.end_id,
    use_cache: bool = False,
    k_cache: Optional[torch.FloatTensor] = None,
    v_cache: Optional[torch.FloatTensor] = None,
    last_token_hidden_states: Optional[torch.FloatTensor] = None,
    use_gpt_decoder_ops: bool = True,
    skip_end_tokens: bool = True,
    detokenize: bool = True,
    echo: bool = False):
    """
    generate output for batch input contexts.

    return_cum_log_probs:
        Whether to compute the cumulative log probsbility of sentences.
        0: do not return the cumulative log probs 
        1: return the cumulative log probs of generated sequences
        2: return the cumulative log probs of sequences
    use_cache:
        if use_cache,k_cahce/v_cache/last_token_hidden_states need to set and in the same device of model weights.
    """
    if use_cache:
        assert use_cache and k_cache is not None and v_cache is not None and last_token_hidden_states is not None, 'at least one of k_cahe/v_cache/last_token_hiddden_states is None'
    batch_size = len(contexts)
    assert batch_size <= model_config.max_batch_size, 'please make sure batch size less/equal model_config.max_batch_size'

    return_output_length = return_cum_log_probs > 0
    if model_config.enable_random_seed:
        random_seed_tensor = torch.randint(0, 10000, size=[batch_size], dtype=torch.int64)
    else:
        random_seed_tensor = torch.zeros([batch_size], dtype=torch.int64)

    bad_words_list=None
    if model_config.banned_words:
        batch_banned_words = model_config.banned_words.split("|")
        banned_words = [[banned_words_for_batch] for banned_words_for_batch in batch_banned_words]
        bad_words_list = torch.tensor(word_list.to_word_list_format(banned_words, tokenizer)).to("cuda")
    # print('bad words list: ', bad_words_list)
    repetition_penalty_vec = None if repetition_penalty == 1. else repetition_penalty * torch.ones(batch_size, dtype=torch.float32)
    presence_penalty_vec   = None if presence_penalty == 0. else presence_penalty * torch.ones(batch_size, dtype=torch.float32)

    infer_decode_args = dict(
        beam_width=num_beams,
        top_k=top_k * torch.ones(batch_size, dtype=torch.int32),
        top_p=top_p * torch.ones(batch_size, dtype=torch.float32),
        temperature=temperature * torch.ones(batch_size, dtype=torch.float32),
        repetition_penalty=repetition_penalty_vec,
        presence_penalty=presence_penalty_vec,
        beam_search_diversity_rate=beam_search_diversity_rate * torch.ones(batch_size, dtype=torch.float32),
        len_penalty=len_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
        bad_words_list=bad_words_list,
        min_length=min_length * torch.ones(size=[batch_size], dtype=torch.int32),
        random_seed=random_seed_tensor,
        use_cache=use_cache,
        k_cache=k_cache,
        v_cache=v_cache,
        last_token_hidden_states=last_token_hidden_states,
    )

    # tokenize context
    max_input_tokens = max_input_tokens or model_config.max_seq_len - max_gen_tokens - 5
    inputs, start_lengths = tokenizer.batch_encode(contexts, max_input_len=max_input_tokens, max_gen_len=max_gen_tokens)
    # import pdb;pdb.set_trace()
    if not model_config.use_gpt_decoder_ops:
        def gpt_generate_fn():
            tokens_batch = gpt(start_ids=inputs['input_ids'],
                                # position_ids=inputs['position_ids'],
                                # block_position_ids=inputs['block_position_ids'],
                                # attention_mask=inputs['attention_mask'],
                               start_lengths=start_lengths,
                               output_len=max_gen_tokens,
                               return_output_length=return_output_length,
                               return_cum_log_probs=return_cum_log_probs,
                               **infer_decode_args)
            return tokens_batch
    else:
        def gpt_generate_fn():
            # print(f"input_token_ids {inputs['input_ids']}")
            # print(f"position_ids {inputs['position_ids']}")
            # print(f"block_position_ids {inputs['block_position_ids']}")
            # print(f"input_lengths {start_lengths}")
            # print(f"gen_length {output_len}")
            # print(f"eos_token_id {end_id}")
            # print(f"return_output_length {return_output_length}")
            # print(f"return_log_probs {return_cum_log_probs}")
            # print(f"infer {infer_decode_args}")
            output_dict = gpt.generate(input_token_ids=inputs['input_ids'],
                                position_ids=inputs['position_ids'],
                                block_position_ids=inputs['block_position_ids'],
                                attention_mask=inputs['attention_mask'],
                                       input_lengths=start_lengths,
                                       gen_length=max_gen_tokens,
                                       eos_token_id=end_id,
                                       return_output_length=return_output_length,
                                       return_log_probs=return_cum_log_probs,
                                       **infer_decode_args)
            return output_dict

    # Generate tokens.
    gen_outputs = gpt_generate_fn()

    if rank == 0:
        if not use_gpt_decoder_ops:
            if return_cum_log_probs > 0:
                tokens_batch, _, cum_log_probs = gen_outputs
            else:
                tokens_batch, cum_log_probs = gen_outputs, None
        else:
            tokens_batch = gen_outputs['output_token_ids']
            cum_log_probs = gen_outputs['cum_log_probs'] if return_cum_log_probs > 0 else None
        if cum_log_probs is not None:
            print('[INFO] Log probs of sentences:', cum_log_probs)

        outputs = []
        tokens_batch = tokens_batch.cpu().numpy()
        for i, (context, tokens) in enumerate(zip(contexts, tokens_batch)):
            for beam_id in range(num_beams):
                token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                if skip_end_tokens:
                    token = token[token != end_id]
                # print('output token: ', token)
                output = tokenizer.decode(token) if detokenize else ' '.join(str(t) for t in token.tolist())
                outputs.append(output)
                if echo:
                    print(f'[INFO] batch {i}, beam {beam_id}:\n[Context]\n{context}\n\n[Output]\n{output}\n')
    return outputs

