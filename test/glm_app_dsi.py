"""
DeepSpeed Inference 
"""
import sys
sys.path.append('glm_10b_chinese')

import asyncio

from sanic import Sanic
from sanic.response import json
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import deepspeed

from modeling_glm import GLMForConditionalGeneration
from model import generate

sema = asyncio.Semaphore(1)

app = Sanic("glm-10b")


@app.listener("before_server_start")
def init(sanic, loop):
    concurrency_per_work = 1
    app.ctx.sem = asyncio.Semaphore(concurrency_per_work, loop=loop)


torch.cuda.set_device(0)

tokenizer = AutoTokenizer.from_pretrained("BAAI/glm-10b-chinese", trust_remote_code=True)
# model = AutoModelForSeq2SeqLM.from_pretrained("BAAI/glm-10b", trust_remote_code=True)
# model.load_state_dict(torch.load('./blocklm-10b-chinese/mp_rank_00_model_states.pt'),strict=False)
# model = GLMForConditionalGeneration.from_pretrained('glm_10b_chinese')
model = GLMForConditionalGeneration.from_pretrained('models/character_20230102_1055')
model = model.half().cuda()

app.ctx.model = model
app.ctx.tokenizer = tokenizer


class ModelInput(BaseModel):
    text: str
    out_seq_length: int = 64
    min_gen_length: int = 64
    sampling_strategy: str = "BeamSearchStrategy"
    num_beams: int = 2
    length_penalty: int = 1
    no_repeat_ngram_size: int = 3
    temperature: float = 0.7
    topk: int = 1
    topp: int = 0


@app.post("/glm10b/complete")
async def handler(request):
    input = ModelInput(**request.json)
    async with request.app.ctx.sem:
        output = await generate(
            **input.dict(exclude={"id"}),
            model=request.app.ctx.model,
            tokenizer=request.app.ctx.tokenizer
        )
    return json({"output": output})


if __name__ == "__main__":
    app.run(host="0.0.0.0",
            port=4000,
            workers=1,
            debug=True)
