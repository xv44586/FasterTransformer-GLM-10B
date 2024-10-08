import oneflow as flow
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from projects.GLM.tokenizer.glm_tokenizer import GLMGPT2Tokenizer, GLMChineseTokenzier
from libai.utils import distributed as dist
from projects.GLM.configs.glm_inference import cfg
from projects.GLM.modeling_glm import GLMForConditionalGeneration
from projects.GLM.utils.glm_loader import GLMLoaderHuggerFace
from omegaconf import DictConfig
import os
import shutil

from functools import wraps
from time import time
totals = []
def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            global totals
            totals.append(end_)
            # print(f"Total execution time: {end_ if end_ > 0 else 0} ms")
    return _time_it


# tokenizer = AutoTokenizer.from_pretrained("BAAI/glm-10b-chinese", trust_remote_code=True)
# tokenizer.save_pretrained("./tokenizer")

tokenizer = GLMChineseTokenzier.from_pretrained("./tokenizer", trust_remote_code=True)
# input_ids = tokenizer.encode(
#     [
#         "测试 [MASK]"
#     ],
#     return_tensors="of",
# )
# print(f"input_ids: {input_ids}")
# inputs = {"input_ids": input_ids, "attention_mask": flow.ones(input_ids.size())}
# inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)

sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
placement = dist.get_layer_placement(0)

dist.set_device_type("cpu")
loader = GLMLoaderHuggerFace(GLMForConditionalGeneration, cfg, "../models/character_20230102_1055")
model = loader.load()
model = model.half().cuda()

# dist.set_device_type("cuda")
# outputs = model.generate(
#     inputs=inputs['input_ids'].to_global(sbp=sbp, placement=placement),
#     position_ids=inputs['position_ids'].to_global(sbp=sbp, placement=placement),
#     generation_attention_mask=inputs['generation_attention_mask'].to_global(sbp=sbp, placement=placement).half(),
#     max_length=32
# )
# res = tokenizer.decode(outputs[0])
# print(res)

dist.set_device_type("cuda")

@measure
def generate():

    input_ids = tokenizer.encode(
    [
        # text,
        # "[MASK]"
        "李⽩，字太⽩，号⻘莲居⼠，⼜号“谪仙⼈”，唐代伟⼤的浪漫主义诗⼈，被后⼈誉为“诗仙”。\\n我：今天我们穿越时空连线李⽩，请问李⽩你爱喝酒吗？\\n李⽩：当然。花间⼀壶酒，独酌⽆相亲。举杯邀明⽉，对影成三⼈。\\n我：你都去过哪些地方？\\n李白：[gMASK]",
    ],
    return_tensors="of",
    )


    # print(f"input_ids: {input_ids}")
    inputs = {"input_ids": input_ids, "attention_mask": flow.ones(input_ids.size())}
    inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)

    outputs = model.generate(
        inputs=inputs['input_ids'].to_global(sbp=sbp, placement=placement),
        position_ids=inputs['position_ids'].to_global(sbp=sbp, placement=placement),
        generation_attention_mask=inputs['generation_attention_mask'].to_global(sbp=sbp, placement=placement).half(),
        max_new_tokens=32,
        # num_beams=1,
        # top_k=1,
        # top_p =0,
        # do_sample=True,
        # temperature=0.7,
        use_cache=False,
        # length_penalty=1,
        # no_repeat_ngram_size=3,
        # eos_token_id=tokenizer.eop_token_id
    )
    res = tokenizer.decode(outputs[0])
    # print(res)



if __name__ == "__main__":
    # texts = ["我说:[MASK]", "测试:[MASK]"]
    for i in range(101):
        generate()

    print(f"totals {totals}")
    total = sum(totals[1:])
    print(f"average: {total/100} ms")