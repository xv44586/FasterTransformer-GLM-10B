import sys
sys.path.append('glm_10b_chinese')


import torch
torch.cuda.set_device(0)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from modeling_glm import GLMForConditionalGeneration
tokenizer = AutoTokenizer.from_pretrained("BAAI/glm-10b-chinese", trust_remote_code=True)
# model = AutoModelForSeq2SeqLM.from_pretrained("BAAI/glm-10b", trust_remote_code=True)
#model.load_state_dict(torch.load('./blocklm-10b-chinese/mp_rank_00_model_states.pt'),strict=False)
# model = GLMForConditionalGeneration.from_pretrained('glm_10b_chinese')
model = GLMForConditionalGeneration.from_pretrained('models/character_20230102_1055')
model = model.half().cuda()


def generator(text, seed, out_seq_length, min_gen_length, sampling_strategy, num_beams, length_penalty, no_repeat_ngram_size, temperature, topk, topp):
    if '[gMASK]' in text:
        mask_id = tokenizer.gmask_token_id
    elif '[MASK]' in text:
        mask_id = tokenizer.mask_token_id
    elif '[sMASK]' in text:
        mask_id = tokenizer.smask_token_id
    else:
        text += '[gMASK]'
        mask_id = tokenizer.gmask_token_id
    inputs = tokenizer(text, return_tensors="pt")
    print(inputs)

    inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
    inputs = {key: value.cuda() for key, value in inputs.items()}
    inputs["generation_attention_mask"] = inputs["generation_attention_mask"].half()
    if sampling_strategy == 'BeamSearchStrategy':
        outputs = model.generate(**inputs, max_new_tokens=out_seq_length, min_length=min_gen_length, eos_token_id=tokenizer.eop_token_id, 
                            num_beams=num_beams, length_penalty=length_penalty, no_repeat_ngram_size=no_repeat_ngram_size)
    else:
        outputs = model.generate(**inputs, max_new_tokens=out_seq_length, min_length=min_gen_length, eos_token_id=tokenizer.eop_token_id, 
                            temperature=temperature, top_k=topk, top_p=topp)
                            
    output = tokenizer.decode(outputs[0].tolist())
    print(output)
    return output 


import gradio as gr

text = ["""李⽩，字太⽩，号⻘莲居⼠，⼜号“谪仙⼈”，唐代伟⼤的浪漫主义诗⼈，被后⼈誉为“诗仙”。
我：今天我们穿越时空连线李⽩，请问李⽩你爱喝酒吗？
李⽩：当然。花间⼀壶酒，独酌⽆相亲。举杯邀明⽉，对影成三⼈。
我：你都去过哪些地方？
李白：[gMASK]"""]

en_fil = ['The Starry Night is an oil-on-canvas painting by [MASK] in June 1889.']
en_gen = ['Eight planets in solar system are [gMASK]']
ch_fil = ['凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。']
ch_gen = ['三亚位于海南岛的最南端,是中国最南部的热带滨海旅游城市 [gMASK]']
en_to_ch = ['Pencil in Chinese is [MASK].']
ch_to_en = ['"我思故我在"的英文是"[MASK]"。']

examples = [en_fil, en_gen, ch_fil, ch_gen, en_to_ch, ch_to_en, text]

with gr.Blocks() as demo:
    gr.Markdown(
        """
        GLM-10B uses two different mask tokens: `[MASK]` for short blank filling and `[gMASK]` for left-to-right long text generation. When the input does not contain any MASK token, `[gMASK]` will be automatically appended to the end of the text. We recommend that you use `[MASK]` to try text fill-in-the-blank to reduce wait time (ideally within seconds without queuing).
        """)

    with gr.Row():
        with gr.Column():
            model_input = gr.Textbox(lines=7, placeholder='Input something in English or Chinese', label='Input')
            with gr.Row():
                gen = gr.Button("Generate")
                clr = gr.Button("Clear")
                
        outputs = gr.Textbox(lines=7, label='Output')
            
    gr.Markdown(
        """
        Generation Parameter
        """)
    with gr.Row():
        with gr.Column():
            seed = gr.Slider(maximum=100000, value=1234, step=1, label='Seed')
            out_seq_length = gr.Slider(maximum=256, value=128, minimum=32, step=1, label='Output Sequence Length')
        with gr.Column():
            min_gen_length = gr.Slider(maximum=64, value=0, step=1, label='Min Generate Length')
            sampling_strategy = gr.Radio(choices=['BeamSearchStrategy', 'BaseStrategy'], value='BeamSearchStrategy', label='Search Strategy')

    with gr.Row():
        with gr.Column():
            # beam search
            gr.Markdown(
                """
                BeamSearchStrategy
                """)
            num_beams = gr.Slider(maximum=4, value=2, minimum=1, step=1, label='Number of Beams')
            length_penalty = gr.Slider(maximum=1, value=1, minimum=0, label='Length Penalty')
            no_repeat_ngram_size = gr.Slider(maximum=5, value=3, minimum=1, step=1, label='No Repeat Ngram Size')
        with gr.Column():
            # base search
            gr.Markdown(
                """
                BaseStrategy
                """)
            temperature = gr.Slider(maximum=1, value=0.7, minimum=0, label='Temperature')
            topk = gr.Slider(maximum=40, value=1, minimum=0, step=1, label='Top K')
            topp = gr.Slider(maximum=1, value=0, minimum=0, label='Top P')
        
    inputs = [model_input, seed, out_seq_length, min_gen_length, sampling_strategy, num_beams, length_penalty, no_repeat_ngram_size, temperature, topk, topp]
    gen.click(fn=generator, inputs=inputs, outputs=outputs)
    clr.click(fn=lambda value: gr.update(value=""), inputs=clr, outputs=model_input)
    
    gr_examples = gr.Examples(examples=examples, inputs=model_input)


demo.launch(share=True, server_name='0.0.0.0')
