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
sys.path.append('../examples/pytorch/glm/gpt_glm_tokenizer')

import torch
from torch.nn.utils.rnn import pad_sequence

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../../.."))
import examples.pytorch.gpt.utils.gpt_token_encoder as encoder
from examples.pytorch.gpt.utils import comm
from examples.pytorch.gpt.utils import gpt_decoder
from examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT

from utils import word_list

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_num', type=int, default=24,
                        help='number of layers')
    parser.add_argument('--input_len', type=int, default=1,
                        help='input sequence length to generate.')
    parser.add_argument('--output_len', type=int, default=32,
                        help='output sequence length to generate.')
    parser.add_argument('--head_num', type=int, default=16,
                        help='head number')
    parser.add_argument('--size_per_head', type=int, default=64,
                        help='size per head')
    parser.add_argument('--vocab_size', type=int, default=50048,
                        help='vocab size')
    parser.add_argument('--beam_width', type=int, default=1,
                        help='beam width for beam search. Using sampling when beam width is 1.')
    parser.add_argument('--top_k', type=int, default=1,
                        help='top k candidate num')
    parser.add_argument('--top_p', type=float, default=0.,
                        help='top p probability threshold')
    parser.add_argument('--temperature', type=float, default=1.,
                        help='temperature')
    parser.add_argument('--len_penalty', type=float, default=0.,
                        help='len_penalty')
    parser.add_argument('--beam_search_diversity_rate', type=float, default=0.,
                        help='beam_search_diversity_rate')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--ckpt_path', type=str, default='../models/huggingface-models/c-model/gpt2-glm/1-gpu/',
                        help='path to the checkpoint file.')
    parser.add_argument('--lib_path', type=str, default='./lib/libth_transformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    # parser.add_argument('--vocab_file', type=str, default="../models/gpt2-vocab.json",
                        # help='vocabulary file.')
    # parser.add_argument('--merges_file', type=str, default="../models/gpt2-merges.txt",
                        # help='merges file.')
    parser.add_argument('--start_id', type=int, default=50006,  # <|startofpiece|>
                        help='start token id.')
    parser.add_argument('--end_id', type=int, default=50000,  # <|endoftext|>
                        help='end token id.')
    parser.add_argument('--max_batch_size', type=int, default=8,
                        help='max batch size.')
    parser.add_argument('--repetition_penalty', type=float, default=1.,
                        help='repetition penalty')
    parser.add_argument('--presence_penalty', type=float, default=0.,
                        help='presence penalty. Similar to repetition, but addive rather than multiplicative.')
    parser.add_argument('--min_length', type=int, default=0,
                        help='A minimum number of tokens to generate')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='max sequence length for position embedding table.')
    parser.add_argument('--inference_data_type', '--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--time', action='store_true',
                        help='whether or not to measure time elapsed.')
    parser.add_argument('--sample_input_file', type=str, default=None,
                        help='path to sample input file. If not set, it runs with no context inputs.')
    parser.add_argument('--sample_output_file', type=str, default=None,
                        help='path to sample output file.')
    parser.add_argument('--enable_random_seed', action='store_true',
                        help='is use the random seed for sentences in a batch.')
    parser.add_argument('--skip_end_tokens', dest='skip_end_tokens', action='store_true',
                        help='Whether to remove or not end tokens in outputs.')
    parser.add_argument('--no_detokenize', dest='detokenize', action='store_false',
                        help='Skip detokenizing output token ids.')
    parser.add_argument('--use_jieba_tokenizer', action='store_true',
                        help='use JiebaBPETokenizer as tokenizer.')
    parser.add_argument('--int8_mode', type=int, default=0, choices=[0, 1],
                        help='The level of quantization to perform.'
                             ' 0: No quantization. All computation in data_type'
                             ' 1: Quantize weights to int8, all compute occurs in fp16/bf16. Not supported when data_type is fp32')
    parser.add_argument(
        '--weights_data_type',
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help='Data type of FT checkpoint weights',
    )
    parser.add_argument('--return_cum_log_probs', type=int, default=0, choices=[0, 1, 2],
                        help='Whether to compute the cumulative log probsbility of sentences.'
                             ' 0: do not return the cumulative log probs '
                             ' 1: return the cumulative log probs of generated sequences'
                             ' 2: return the cumulative log probs of sequences')
    parser.add_argument('--shared_contexts_ratio', type=float, default=1.0,
                        help='Triggers the shared context optimization when'
                             'compact_size <= shared_contexts_ratio * batch_size'
                             'A value of 0.0 deactivate the optimization')
    parser.add_argument('--banned_words',
        type=str,
        default="",
        help='A comma separated list of tokens that should never be generated. Everything between the commas will'
             ' be tokenized and converted to token ids that will be banned.'
             ' Note that spaces before and after commas are included in tokenization.'
             ' An example highlighting this importance is that "the" and " the" are'
             ' two separate tokens some vocabularies.'
             ' Therefore, do ban a certain phrase, we would need to specify all tokens'
             ' in the vocabulary that include the phrase.'
             ' Example use: --banned_words "the, the,a,boy". This will ban the tokens "the", " the", "a" and "boy".'
             ' We can also use a pipe "|" to ban different tokens for different sentences in a batch.'
             ' Example: --banned_words "the, the|a,boy" will ban the tokens "the" and " the" in output sentence 1 and'
             ' ban the tokens "a" and "boy" in output sentence 2. When using this mode, we must specify a set of tokens to ban'
             ' for each sentence in the batch.',
    )
    parser.add_argument('--use_gpt_decoder_ops', action='store_true',
                        help='Use separate decoder FT operators instead of end-to-end model op.')
    parser.add_argument('--use_cache', default='', help='k/v/last_token_hidden_states cache path')
    args = parser.parse_args()

    ckpt_config = configparser.ConfigParser()
    ckpt_config_path = os.path.join(args.ckpt_path, 'config.ini')
    if os.path.isfile(ckpt_config_path):
        ckpt_config.read(ckpt_config_path)
    if 'gpt' in ckpt_config.keys():
        for args_key, config_key, func in [
            ('layer_num', 'num_layer', ckpt_config.getint),
            ('max_seq_len', 'max_pos_seq_len', ckpt_config.getint),
            ('weights_data_type', 'weight_data_type', ckpt_config.get),
        ]:
            if config_key in ckpt_config['gpt'].keys():
                prev_val = args.__dict__[args_key]
                args.__dict__[args_key] = func('gpt', config_key)
                print('Loading {} from config.ini,    previous: {},    current: {}'.format(
                    args_key, prev_val, args.__dict__[args_key]))
            else:
                print('Not loading {} from config.ini'.format(args_key))
        for key in ['head_num', 'size_per_head', 'tensor_para_size']:
            if key in args.__dict__:
                prev_val = args.__dict__[key]
                args.__dict__[key] = ckpt_config.getint('gpt', key)
                print('Loading {} from config.ini,    previous: {},    current: {}'.format(
                    key, prev_val, args.__dict__[key]))
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

    layer_num = args.layer_num
    output_len = args.output_len
    head_num = args.head_num
    size_per_head = args.size_per_head
    vocab_size = args.vocab_size
    beam_width = args.beam_width
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    len_penalty = args.len_penalty
    beam_search_diversity_rate = args.beam_search_diversity_rate
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    start_id = args.start_id
    end_id = args.end_id
    max_batch_size = args.max_batch_size
    max_seq_len = args.max_seq_len
    repetition_penalty = args.repetition_penalty
    presence_penalty = args.presence_penalty
    min_length = args.min_length
    weights_data_type = args.weights_data_type
    return_cum_log_probs = args.return_cum_log_probs
    return_output_length = return_cum_log_probs > 0
    shared_contexts_ratio = args.shared_contexts_ratio
    use_cache = args.use_cache

    print('\n=================== Arguments ===================')
    for k, v in vars(args).items():
        print(f'{k.ljust(30, ".")}: {v}')
    print('=================================================\n')

    # if args.use_jieba_tokenizer:
    #     from examples.pytorch.gpt.utils.tokenizer import JiebaBPETokenizer
    #     enc = JiebaBPETokenizer(args.vocab_file)
    # else:
    #     enc = encoder.get_encoder(args.vocab_file, args.merges_file)
    from tokenization_glm import GLMChineseTokenizer
    tokenizer = GLMChineseTokenizer.from_pretrained('../examples/pytorch/glm/gpt_glm_tokenizer')
    
    def encode(txt):
        inputs = tokenizer(txt, return_tensors='pt')
        inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=max_seq_len)  # generate max seq len position ids
        return inputs

    def batch_encode(txt_list):
        batch_encoded = [encode(txt) for txt in txt_list]
        if len(batch_encoded) == 1:
            batch_encoded = batch_encoded[0]
            batch_encoded['input_ids'] = batch_encoded['input_ids'].to(torch.int32)
            batch_encoded['attention_mask'] = batch_encoded['attention_mask'][:,:max_seq_len, :max_seq_len]
            # print(batch_encoded, batch_encoded['input_ids'].shape[1])

            return batch_encoded, torch.IntTensor([batch_encoded['input_ids'].shape[1] ])
        batch_inputs = {}
        input_ids = [encoded['input_ids'][0] for encoded in batch_encoded]
        start_lengths = torch.IntTensor([ids.size()[0]  for ids in input_ids])
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.eos_token_id)
        # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)  # use <unk> token
        batch_inputs['input_ids'] = input_ids.to(torch.int32)

        position_ids = [encoded['position_ids'][0, :max_seq_len] for encoded in batch_encoded]
        position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)
        batch_inputs['position_ids'] = position_ids

        block_position_ids = [encoded['block_position_ids'][0, :max_seq_len] for encoded in batch_encoded]
        block_position_ids = pad_sequence(block_position_ids, batch_first=True, padding_value=0)
        batch_inputs['block_position_ids'] = block_position_ids

        attention_mask = [encoded['attention_mask'][0, :max_seq_len, :max_seq_len] for encoded in batch_encoded]
        # import pdb;pdb.set_trace()
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        batch_inputs['attention_mask'] = attention_mask

        # print('inputs: ', txt_list)
        print('start lengths: ', start_lengths)
        # print('batch inputs: ', batch_inputs['input_ids'])
        return batch_inputs, start_lengths


    # torch.manual_seed(0)

    comm.initialize_model_parallel(args.tensor_para_size, args.pipeline_para_size)
    rank = comm.get_rank()
    device = comm.get_device()

    # Inputs
    contexts = []
    if args.sample_input_file:  # conditional case
        if args.sample_input_file.endswith('json'):
            import json
            contexts = json.load(open(args.sample_input_file))
            contexts = [c['prompt'] for c in contexts]
        else:
            with open(args.sample_input_file, "r") as f:
                contexts = f.read().splitlines()
        batch_size = min(len(contexts), max_batch_size)
        contexts = contexts[-batch_size:]
        # start_ids = [torch.tensor(enc.encode(c), dtype=torch.int32, device=device) for c in contexts]
        print(f"contexts {contexts}")
        inputs, start_lengths = batch_encode(contexts)
    else:  # unconditional case
        batch_size = max_batch_size
        # contexts = ['<|endoftext|>'] * batch_size
        # start_ids = [torch.IntTensor([end_id for _ in range(args.input_len)])] * batch_size
        contexts = ['[gMASK]'] * batch_size
        inputs, start_lengths = batch_encode(contexts)
    torch.save(start_lengths, 'start_lengths.pt')
    torch.save(inputs['input_ids'], 'input_ids.pt')
    torch.save(inputs['position_ids'], 'position_ids.pt')
    torch.save(inputs['block_position_ids'], 'block_position_ids.pt')
    # input cache
    k_cache = None
    v_cache = None
    last_token_hidden_states = None
    if use_cache:
        k_cache = torch.load(f'{use_cache}/k_cache.pt').to('cuda')
        v_cache = torch.load(f'{use_cache}/v_cache.pt').to('cuda')
        # [layer_nums, batch_size, head_num, 8?, seq_len, 8?]
        # k_cache = k_cache[:, :, :, :, :max_input_length, :]
        # v_cache shape: [layer_nums, batch_size, head_num, seq_len, size_per_head]
        # v_cache = v_cache[:, :, :, :max_input_length, :]
        last_token_hidden_states = torch.load(f'{use_cache}/last_token_hidden_states.pt').to('cuda')

    # start_lengths = [len(ids) for ids in start_ids]
    # start_lengths = []

    # start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
    # start_lengths = torch.IntTensor(start_lengths)

    # import pdb;pdb.set_trace()
    # Prepare model.
    if not args.use_gpt_decoder_ops:
        gpt = ParallelGPT(head_num, size_per_head, vocab_size, start_id, end_id,
                          layer_num, max_seq_len, tensor_para_size, pipeline_para_size,
                          lib_path=args.lib_path, inference_data_type=args.inference_data_type,
                          int8_mode=args.int8_mode, weights_data_type=weights_data_type,
                          shared_contexts_ratio=shared_contexts_ratio,
                          gpt_with_moe=gpt_with_moe,
                          expert_num=expert_num,
                          moe_k=moe_k,
                          moe_layer_index=moe_layer_index)
        if not gpt.load(ckpt_path=args.ckpt_path):
            print("[WARNING] Checkpoint file not found. Model loading is skipped.")
    else:
        assert moe_layer_index == []
        gpt = gpt_decoder.Gpt(
            num_heads=head_num,
            size_per_head=size_per_head,
            num_layers=layer_num,
            vocab_size=vocab_size,
            start_id=start_id,
            end_id=end_id,
            tensor_para_size=tensor_para_size,
            pipeline_para_size=pipeline_para_size,
            lib_path = args.lib_path,
            max_seq_len=max_seq_len,
            int8_mode=args.int8_mode,
            weights_data_type=args.weights_data_type)
        gpt.load(args.ckpt_path, args.inference_data_type)

    if args.enable_random_seed:
        random_seed_tensor = torch.randint(0, 10000, size=[batch_size], dtype=torch.int64)
    else:
        random_seed_tensor = torch.zeros([batch_size], dtype=torch.int64)
    print('random seed:', random_seed_tensor)

    bad_words_list=None
    if args.banned_words:
        batch_banned_words = args.banned_words.split("|")
        banned_words = [[banned_words_for_batch] for banned_words_for_batch in batch_banned_words]
        bad_words_list = torch.tensor(word_list.to_word_list_format(banned_words, tokenizer)).to("cuda")
    # print('bad words list: ', bad_words_list)
    repetition_penalty_vec = None if repetition_penalty == 1. else repetition_penalty * torch.ones(batch_size, dtype=torch.float32)
    presence_penalty_vec   = None if presence_penalty == 0. else presence_penalty * torch.ones(batch_size, dtype=torch.float32)

    infer_decode_args = dict(
        beam_width=beam_width,
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

    if not args.use_gpt_decoder_ops:
        def gpt_generate_fn():
            tokens_batch = gpt(start_ids=inputs['input_ids'],
                                # position_ids=inputs['position_ids'],
                                # block_position_ids=inputs['block_position_ids'],
                                # attention_mask=inputs['attention_mask'],
                               start_lengths=start_lengths,
                               output_len=output_len,
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
                                       gen_length=output_len,
                                       eos_token_id=end_id,
                                       return_output_length=return_output_length,
                                       return_log_probs=return_cum_log_probs,
                                       **infer_decode_args)
            return output_dict

    # Generate tokens.
    gen_outputs = gpt_generate_fn()

    if rank == 0:
        if not args.use_gpt_decoder_ops:
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
            for beam_id in range(beam_width):
                token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                if args.skip_end_tokens:
                    token = token[token != end_id]
                # print('output token: ', token)
                output = tokenizer.decode(token) if args.detokenize else ' '.join(str(t) for t in token.tolist())
                outputs.append(output)
                if not args.time:
                    print(f'[INFO] batch {i}, beam {beam_id}:\n[Context]\n{context}\n\n[Output]\n{output}\n')

        if args.sample_output_file:
            with open(args.sample_output_file, "w+") as f:
                outputs = [o.replace("\n", "\\n") for o in outputs]
                f.writelines("\n".join(outputs))

    # Measure inference time.
    if args.time:
        iterations = 10
        for _ in range(iterations):
            gpt_generate_fn()
        time = timeit.default_timer()
        for _ in range(iterations):
            gpt_generate_fn()
        time_elapsed = timeit.default_timer() - time
        print(f'[INFO] GPT time costs: {time_elapsed * 1000 / iterations:.2f} ms')


if __name__ == '__main__':
    main()
