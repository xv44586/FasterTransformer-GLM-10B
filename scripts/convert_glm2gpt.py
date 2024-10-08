from collections import OrderedDict
import os
import torch
from argparse import ArgumentParser


def convert(glm_ckpt, save_path, num_layers):
    glm = torch.load(glm_ckpt, map_location='cpu')
    gpt = OrderedDict()
    # embeddings
    gpt['wte.weight'] = glm['word_embeddings.weight']
    gpt['wpe.weight'] = glm['transformer.position_embeddings.weight']
    gpt['wbpe.weight'] = glm['transformer.block_position_embeddings.weight']
    # transformer blocks
    for idx in range(num_layers):
        gpt['h.%d.ln_1.weight' % idx] = glm['transformer.layers.%d.input_layernorm.weight' % idx]
        gpt['h.%d.ln_1.bias' % idx] = glm['transformer.layers.%d.input_layernorm.bias' % idx]
        gpt['h.%d.attn.c_attn.weight' % idx] = glm['transformer.layers.%d.attention.query_key_value.weight' % idx].T
        gpt['h.%d.attn.c_attn.bias' % idx] = glm['transformer.layers.%d.attention.query_key_value.bias' % idx]
        gpt['h.%d.attn.c_proj.weight' % idx] = glm['transformer.layers.%d.attention.dense.weight' % idx].T
        gpt['h.%d.attn.c_proj.bias' % idx] = glm['transformer.layers.%d.attention.dense.bias' % idx]
        gpt['h.%d.ln_2.weight' % idx] = glm['transformer.layers.%d.post_attention_layernorm.weight' % idx]
        gpt['h.%d.ln_2.bias' % idx] = glm['transformer.layers.%d.post_attention_layernorm.bias' % idx]
        gpt['h.%d.mlp.c_fc.weight' % idx] = glm['transformer.layers.%d.mlp.dense_h_to_4h.weight' % idx].T
        gpt['h.%d.mlp.c_fc.bias' % idx] = glm['transformer.layers.%d.mlp.dense_h_to_4h.bias' % idx]
        gpt['h.%d.mlp.c_proj.weight' % idx] = glm['transformer.layers.%d.mlp.dense_4h_to_h.weight' % idx].T
        gpt['h.%d.mlp.c_proj.bias' % idx] = glm['transformer.layers.%d.mlp.dense_4h_to_h.bias' % idx]
    
    # final layer
    gpt['ln_f.weight'] = glm['transformer.final_layernorm.weight']
    gpt['ln_f.bias'] = glm['transformer.final_layernorm.bias']

    output_path = os.path.join(save_path, 'pytorch_model.bin')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(gpt, output_path)
    print('done.')


# convert('./glm_10b_chinese/pytorch_model.bin', 'gpt_glm', 48)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--glm_ckpt', default='./glm_10b_chinese/pytorch_model.bin')
    parser.add_argument('--save_dir', default='gpt_glm_10b_chinese', help='transformers GPT2-style weights save dir')
    parser.add_argument('--num_layers', default=48, help='transformer layer num')
    args = parser.parse_args()
    print('args: ', args)
    convert(args.glm_ckpt, args.save_dir, args.num_layers)