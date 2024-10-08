# 备份留念
当前repo 是整理后的早期基于FasterTransformer 搭建glm-10b-chinese 模型server 相关代码

===========================================================================

# glm-10b-chinese server 
glm-10b-chinese 目前主要有两种server 形式：基于transformers版本与基于FasterTransformer版本。

## transformers 版本
transformers 版本代码主要也有两份：原始版本及优化版本 和 基于GPT2 重构版。

### 原始版本
原始版本代码写的很乱，有很多无关代码，cache 部分实现也有问题，我进行了优化，对应所有代码都在*glm_10b_chinese* 目录下。

- glm-10b-chinese
  - modeling_glm.py  # 原始版本
  - modeling_glm_cache.py # 优化cache 部分
  - modeling_glm_cache_pruning.py  # 删除无用分支/代码
  - modeling_glm_cache_pruning_prompt.py  # cache + prompt-cache 版本

### 基于GPT2重构版
为了在其他server 框架(onnx/FasterTransformer) 中运行，对glm-10b-chinese 进行了重构，对比其与GPT2 的差异。对应代码在*gpt_glm_10b_chinese*目录下
注意，对应的config 也不同了，copy 时记得完整拷贝。

- gpt_glm_10b_chinese
  - modeling_gpt2.py  # 基于GPT-2 重构版

## weights 转换
将原始transformers 版本的glm-10b-chinese 的ckpt 转换为重构后的基于transformers/GPT-2 的ckpt
```bash
python scripts/convert_glm_ckpt_to_gpt.py --glm-ckpt ./glm_10b_chinese/pytorch_model.bin --save_dir gpt_glm --num_layers 48
``` 

## FasterTransformer server
FasterTransformer 需要的环境比较难装，这里使用docker image。

### pull docker image
You can choose the tensorflow version and python version you want. Here, we list some possible images:
To achieve best performance, we recommend to use the latest image. For example, running image `nvcr.io/nvidia/pytorch:22.09-py3` by

```bash
# nvidia-docker run -ti --shm-size 5g --rm nvcr.io/nvidia/pytorch:22.09-py3 bash
sudo docker run -itd --rm --gpus all --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 --net=host --ipc=host --privileged -v /mnt:/mnt -v /data:/data --name glm nvcr.io/nvidia/pytorch:22.09-py3

git clone https://github.com/NVIDIA/FasterTransformer.git
mkdir -p FasterTransformer/build
cd FasterTransformer/build
git submodule init && git_proxy=http://commercial-proxy-int.trafficmanager.net:8443 git submodule update
```

### copy code
将glm 相关代码移动到对应位置
```bash
cp -r FasterTransformers_glm/glm FasterTransformer/pytorch/examples
```

### remove/remark build decoder attention mask code
```bash
# src/fastertransformer/models/multi_gpu_gpt/ParalleGpt.cc +1030
            PUSH_RANGE("build decoder attention mask");
            // invokeBuildDecoderAttentionMask(input_attention_mask_,
            //                                 tiled_input_lengths_buf_,
            //                                 nullptr,
            //                                 batch_size * beam_width,
            //                                 max_input_length,
            //                                 0,
            //                                 stream_);
            sync_check_cuda_error();
            POP_RANGE;
```

### build with pytorch
```bash
cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
make -j12
```

### Install required tools
```bash
pip install -r ../examples/pytorch/gpt/requirement.txt -i https://mirrors.cloud.tencent.com/pypi/simple
```

### convert glm weights

```bash
python ../examples/pytorch/gpt/utils/huggingface_gpt_convert.py -i ../../gpt_glm_10b_chinese/ -o  ../models/huggingface-models/c-model/gpt2-glm -i_g 1
```

### test
```bash
python ../examples/pytorch/glm/multi_gpu_glm_example.py  --ckpt_path ../models/huggingface-models/c-model/character_20230102/1-gpu/ --sample_input_file prompts4.json  --max_batch_size 4 --use_gpt_decoder_ops  --inference_data_type bf16 --sample_output_file output.txt
```

