from pydantic import BaseSettings


class ModelConfig(BaseSettings):
    """All settings for glm."""
    layer_num: int = 48
    input_len: int = 1
    output_len: int = 256
    head_num: int = 64
    size_per_head: int = 64
    vocab_size: int = 50048
    beam_width: int = 1
    top_k: int = 1
    top_p: float = 0.0
    temperature: float = 1.0
    len_penalty: float = 1.0
    beam_search_diversity_rate: float = 0.0
    tensor_para_size: int = 1
    pipeline_para_size: int = 1
    ckpt_path: str = "/data/project/BusinessSolution-Morpheus/private/src/glm-server/FasterTransformer/models/huggingface-models/c-model/gpt2-glm/1-gpu/"
    lib_path: str = "/data/project/BusinessSolution-Morpheus/private/src/glm-server/FasterTransformer/build/lib/libth_transformer.so"
    token_path: str = "/data/project/BusinessSolution-Morpheus/private/src/glm-server/FasterTransformer/examples/pytorch/glm/gpt_glm_tokenizer"
    start_id: int = 50006
    end_id: int = 50007
    max_batch_size: int = 5
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    min_length: int = 0
    max_seq_len: int = 1025
    inference_data_type: str = "bf16"
    measure_time: bool = False
    sample_input_file: str = "input.json"
    sample_output_file: str = "output.txt"
    enable_random_seed: bool = False
    skip_end_tokens: bool = False
    detokenize: bool = True
    int8_mode: bool = False
    weights_data_type: str = "fp32"
    return_cum_log_probs: int = 0 # 0, 1, 2
    shared_contexts_ratio: float = 1.0
    banned_words: str = ""
    use_gpt_decoder_ops: bool = True

    class Config:
        case_sensitive = False
        env_prefix = ""

model_config = ModelConfig()
