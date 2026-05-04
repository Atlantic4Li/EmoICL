import transformers
import torch


def load_i2t_model(engine, args=None):
    if engine == 'otter-llama':
        from otter_ai import OtterForConditionalGeneration
        model = OtterForConditionalGeneration.from_pretrained(
            "/data1/yq_log/IntenICL/huggingface/luodian/OTTER-Image-LLaMA7B-LA-InContext",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = model.text_tokenizer
        image_processor = transformers.CLIPImageProcessor()
        processor = image_processor
    elif 'llava-onevision-0.5b' in engine:
        from llava.model.builder import load_pretrained_model as load_llava_model
        tokenizer, model, image_processor, context_len = load_llava_model(
            model_path='/data1/yq_log/IntenICL/huggingface/llava-onevision-qwen2-0.5b-ov',
            model_base=None,
            attn_implementation="flash_attention_2",
            model_name='llava_qwen',
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        processor = image_processor
    elif 'llava-onevision-7b' in engine:
        from llava.model.builder import load_pretrained_model as load_llava_model
        tokenizer, model, image_processor, context_len = load_llava_model(
            model_path='/data1/yq_log/IntenICL/huggingface/llava-onevision-qwen2-7b-ov',
            model_base=None,
            attn_implementation="sdpa",
            device_map="auto",
            model_name='llava_qwen',
            torch_dtype=torch.bfloat16,
        )
        processor = image_processor
    elif engine == 'qwen-vl':
        from transformers.generation import GenerationConfig
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "/data1/yq_log/IntenICL/huggingface/Qwen/Qwen-VL",
            trust_remote_code=True,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "/data1/yq_log/IntenICL/huggingface/Qwen/Qwen-VL",
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        model.generation_config = GenerationConfig.from_pretrained(
            "/data1/yq_log/IntenICL/huggingface/Qwen/Qwen-VL",
            trust_remote_code=True,
        )
        processor = None
    elif engine == 'idefics-9b-instruct':
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        checkpoint = "/data1/yq_log/IntenICL/huggingface/HuggingFaceM4/idefics-9b-instruct"
        model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    else:
        raise NotImplementedError(f"Unsupported engine: {engine}")
    return model, tokenizer, processor
