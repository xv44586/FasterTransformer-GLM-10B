import torch
torch.cuda.set_device(0)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("BAAI/glm-10b", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("BAAI/glm-10b", trust_remote_code=True)
model = model.half().cuda()

text = """李⽩，字太⽩，号⻘莲居⼠，⼜号“谪仙⼈”，唐代伟⼤的浪漫主义
诗⼈，被后⼈誉为“诗仙”。
我：今天我们穿越时空连线李⽩，请问李⽩你爱喝酒吗？
李⽩：当然。花间⼀壶酒，独酌⽆相亲。举杯邀明⽉，对影成三⼈。
我：你都去过哪些地方？
李白：[gMASK]"""

text = '问题：冬天，中国哪座城市最适合避寒？问题描述：能推荐一些国内适合冬天避寒的城市吗？回答用户：旅游爱好者 回答： [gMASK]'

def gen(text, seed, max_length, min_length,num_beams, length_penalty, no_repeat_ngram, temperature, top_p, top_k):
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

    inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512, mask_id=mask_id)
    inputs = {key: value.cuda() for key, value in inputs.items()}
    inputs["generation_attention_mask"] = inputs["generation_attention_mask"].half()
    outputs = model.generate(**inputs, max_new_tokens=max_length, min_length=min_length, eos_token_id=tokenizer.eop_token_id, 
                            num_beams=num_beams, length_penalty=length_penalty, no_repeat_ngram_size=no_repeat_ngram, temperature=temperature,
                            top_p=top_p, top_k=top_k)
                            
    output = tokenizer.decode(outputs[0].tolist())
    print(output)
    return output 

import gradio as gr



examples = [
    ["问题：冬天，中国哪座城市最适合避寒？问题描述：能推荐一些国内适合冬天避寒的城市吗？回答用户：旅游爱好者 回答："],
    ["工业互联网（Industrial Internet）是新一代信息通信技术与工业经济深度融合的新型基础设施、应用模式和工业生态，通过对人、机、物、系统等的全面连接，构建起覆盖全产业链、全价值链的全新制造和服务体系，为工业乃至产业数字化、网络化、智能化发展提供了实现途径，是第四次工业革命的重要基石。[sMASK]它以网络为基础、平台为中枢、数据为要素、安全为保障，既是工业数字化、网络化、智能化转型的基础设施，也是互联网、大数据、人工智能与实体经济深度融合的应用模式，同时也是一种新业态、新产业，将重塑企业形态、供应链和产业链。当前，工业互联网融合应用向国民经济重点行业广泛拓展，形成平台化设计、智能化制造、网络化协同、个性化定制、服务化延伸、数字化管理六大新模式，赋能、赋智、赋值作用不断显现，有力的促进了实体经济提质、增效、降本、绿色、安全发展。"],
]

# demo = gr.Interface(
#     fn=gen,
#     inputs=[gr.inputs.Textbox(lines=5, label="Input Text"), gr.Slider(minimu=0, maximum=10000),
#     outputs=gr.outputs.Textbox(label="Generated Text"),
#     examples=examples
# )
demo = gr.Interface(
    fn=gen,
    inputs=[gr.inputs.Textbox(lines=5, label="Input Text"), 
            gr.Slider(minimu=0, maximum=100000, value=1234, step=1, label='Seed'), 
            gr.Slider(minimu=16, maximum=256, value=32, step=1, label='Output Sequence Length'),
            gr.Slider(minimu=0, maximum=64, value=0, step=1, label='Min Generate Length'),           
            gr.Slider(minimu=1, maximum=4, value=2, step=1, label='Number of Beams'),
            gr.Slider(minimu=0, maximum=1, value=1, step=0.01, label='Length Penalty'),
            gr.Slider(minimu=1, maximum=5, value=3, step=1, label='No Repeat Ngram Size'),
            gr.Slider(maximum=1, value=0.7, step=0.01, label='Temperature'),
            gr.Slider(maximum=40, value=1, step=1, label='Top K'),
            gr.Slider(maximum=1, value=0, step=0.01, label='Top P'),
            ],
    outputs=gr.outputs.Textbox(label="Generated Text"),
    examples=examples
)
demo.launch(share=True, server_name='0.0.0.0')