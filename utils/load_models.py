import transformers
import os
import torch


def load_i2t_model(engine, args=None):
    if engine == 'otter-mpt':
        from otter_ai import OtterForConditionalGeneration
        model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-Image-MPT7B", device_map="cuda", torch_dtype=torch.bfloat16)
        tokenizer = model.text_tokenizer
        image_processor = transformers.CLIPImageProcessor()
        processor = image_processor
    elif engine == 'otter-llama':
        from otter_ai import OtterForConditionalGeneration
        model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-Image-LLaMA7B-LA-InContext", device_map="cuda", torch_dtype=torch.bfloat16)
        tokenizer = model.text_tokenizer
        image_processor = transformers.CLIPImageProcessor()
        processor = image_processor
    elif engine == 'llava16-7b':
        from llava.model.builder import load_pretrained_model as load_llava_model
        tokenizer, model, image_processor, context_len = load_llava_model(model_path='/data0/lxy_data/huggingface/llava-v1.6-vicuna-7b', model_base=None, model_name='llava-v1.6-vicuna-7b', device_map="auto", torch_dtype=torch.bfloat16)
        processor = image_processor
    elif 'llava-onevision-0.5b' in engine:
        from llava.model.builder import load_pretrained_model as load_llava_model
        tokenizer, model, image_processor, context_len = load_llava_model(model_path='/data0/lxy_data/huggingface/llava-onevision-qwen2-0.5b-ov', model_base=None, attn_implementation="flash_attention_2", model_name='llava_qwen', device_map="cuda", torch_dtype=torch.bfloat16)
        processor = image_processor
    elif 'llava-onevision-7b' in engine:
        from llava.model.builder import load_pretrained_model as load_llava_model
        tokenizer, model, image_processor, context_len = load_llava_model(model_path='/data0/lxy_data/huggingface/llava-onevision-qwen2-7b-ov', model_base=None, attn_implementation="flash_attention_2", model_name='llava_qwen', device_map="auto", torch_dtype=torch.bfloat16)
        processor = image_processor
    elif engine == 'qwen-vl-chat':
        from transformers.generation import GenerationConfig
        tokenizer = transformers.AutoTokenizer.from_pretrained("/data0/lxy_data/huggingface/Qwen/Qwen-VL-Chat", trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("/data0/lxy_data/huggingface/Qwen/Qwen-VL-Chat", device_map="auto", 
                                                                  trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        model.generation_config = GenerationConfig.from_pretrained("/data0/lxy_data/huggingface/Qwen/Qwen-VL-Chat", trust_remote_code=True)
        processor = None
    elif engine == 'qwen-vl':
        from transformers.generation import GenerationConfig
        tokenizer = transformers.AutoTokenizer.from_pretrained("/data0/lxy_data/huggingface/Qwen/Qwen-VL", trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("/data0/lxy_data/huggingface/Qwen/Qwen-VL", device_map="auto", trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained("/data0/lxy_data/huggingface/Qwen/Qwen-VL", trust_remote_code=True)
        processor = None
    elif engine == 'internlm-x2':
        model = transformers.AutoModel.from_pretrained('internlm/internlm-xcomposer2-7b', trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
        tokenizer = transformers.AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-7b', trust_remote_code=True)
        model.tokenizer = tokenizer
        processor = None
    elif engine == 'openflamingo':
        from open_flamingo import create_model_and_transforms
        model, processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=4,
        )
        model = model.to(torch.bfloat16).cuda()

    elif engine == 'emu2-chat':
        from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
        tokenizer = transformers.AutoTokenizer.from_pretrained("BAAI/Emu2-Chat")
        with init_empty_weights():
            model = transformers.AutoModelForCausalLM.from_pretrained(
                "BAAI/Emu2-Chat",
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True).eval()
        # adjust according to your device
        device_map = infer_auto_device_map(model, max_memory={0:'38GiB',1:'38GiB',2:'38GiB',3:'38GiB'}, no_split_module_classes=['Block','LlamaDecoderLayer'])
        device_map["model.decoder.lm.lm_head"] = 0

        model = load_checkpoint_and_dispatch(
            model, 
            'path/to/models--BAAI--Emu2-Chat/snapshots/your_snapshot_path',
            device_map=device_map).eval()
        processor = None
    elif engine == 'idefics-9b-instruct':
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        checkpoint = "HuggingFaceM4/idefics-9b-instruct"
        model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == 'idefics-9b':
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        checkpoint = "HuggingFaceM4/idefics-9b"
        model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == 'idefics-80b-instruct':
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
        checkpoint = "HuggingFaceM4/idefics-80b-instruct"
        model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == 'gpt4v':
        model, tokenizer, processor = None, None, None
    else:
        raise NotImplementedError
    return model, tokenizer, processor