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
