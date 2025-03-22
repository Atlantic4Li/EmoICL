import torch
try:
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import tokenizer_image_token
except:
    pass

import os
import time
from PIL import Image
from .ICL_utils import get_task_instruction, format_answer
from .utils import load_image


def ICL_inference(args, engine, dataset, model, tokenizer, query, 
                      n_shot_support, data_path, processor, max_new_tokens):
    task_instruction = get_task_instruction(args)
    img_id = query['image']
    query_images, query_image_paths = load_image(img_id, data_path)
    query_text = query['question']
    if 'qwen-vl' in engine:
        inputs = [{'text': f'You are a helpful assistant. {task_instruction}'}]
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                inputs.append({'image': os.path.join(data_path, image_path)})
            inputs.append({'text': 'User: ' + n_shot_support[i]['question'] + 
                            '\nAssistant: ' + format_answer(n_shot_support[i]['answer'], dataset, query) + '\n'})
        
        for query_image_path in query_image_paths:
            inputs.append({'image': query_image_path})
        inputs.append({'text': 'User: ' + query_text + '\nAssistant:'})
        
        total_inputs = tokenizer.from_list_format(inputs)
        inputs = tokenizer(total_inputs, return_tensors='pt')
        inputs = inputs.to(model.device)
        with torch.no_grad():
            pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
        input_token_len = inputs['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(pred[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    elif 'llava' in engine:
        from llava.conversation import conv_templates
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
        )
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
            input_text += f"{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"
        
        for query_image in query_images:
            images.append(query_image)
            input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
        input_text += f"{query_text}\nAnswer:"
        image_tensor = torch.stack(
                [
                    processor.preprocess(image_file, return_tensors="pt")["pixel_values"][0]
                    for image_file in images
                ]
            )
        image_tensor = image_tensor.half().cuda()
        conv_mode = 'llava_v1' if 'onevision' not in engine else 'qwen_1_5'
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(prompt)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                generated_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    )
        predicted_answers = tokenizer.batch_decode(generated_ids[:, :], skip_special_tokens=True)[0]
    elif 'flamingo' in engine:
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                input_text += "<image>"
            input_text += f"{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}<|endofchunk|>"
        for query_image in query_images:
            images.append(query_image)
            input_text += "<image>"
            
        vision_x = [processor(image).unsqueeze(0) for image in images]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        input_text += f"{query_text}\nAnswer:"
        
        lang_x = tokenizer(
            [input_text],
            return_tensors="pt",
        )
        with torch.no_grad():
            predicted_answers = model.generate(
                vision_x=vision_x.to(torch.bfloat16).cuda(),
                lang_x=lang_x["input_ids"].cuda(),
                attention_mask=lang_x["attention_mask"].cuda(),
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        input_token_len = lang_x['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(predicted_answers[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    elif 'otter' in engine:
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                input_text += "<image>"
            input_text += f"User: {n_shot_support[i]['question']}\nGPT:<answer> {format_answer(n_shot_support[i]['answer'], dataset, query)}<|endofchunk|>"
        for query_image in query_images:
            images.append(query_image)
            input_text += "<image>"
        input_text += f"User: {query_text}\nGPT:<answer>"

        vision_x = processor.preprocess(images, return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        lang_x = model.text_tokenizer(
            [
                input_text,
            ],
            return_tensors="pt",
        )
        bad_words_id = tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
        with torch.no_grad():
            predicted_answers = model.generate(
                vision_x=vision_x.to(model.device),
                lang_x=lang_x["input_ids"].to(model.device),
                attention_mask=lang_x["attention_mask"].to(model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                bad_words_ids=bad_words_id,
            )
        input_token_len = lang_x['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(predicted_answers[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    elif 'internlm-x' in engine:
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                image = Image.open(os.path.join(data_path, image_path)).convert("RGB")
                image = model.vis_processor(image)
                images.append(image)
                input_text += "<ImageHere>"
            input_text += f"{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"
        for query_image in query_images:
            images.append(model.vis_processor(query_image))
            input_text += "<ImageHere>"
        input_text += f"{query_text}\nAnswer:"
        image = torch.stack(images).to(torch.bfloat16).cuda()
        predicted_answers, history = model.chat(tokenizer, query=input_text, image=image, history=[], do_sample=False, max_new_tokens=max_new_tokens)
    elif 'emu2-chat' in engine:
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                input_text += "[<IMG_PLH>]"
            input_text += f"[{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}]."
        for query_image in query_images:
            images.append(query_image)
            input_text += "[<IMG_PLH>]"
        input_text += f"[{query_text}\nAnswer:"
        inputs = model.build_input_ids(
            text=[input_text],
            tokenizer=tokenizer,
            image=images
        )
        
        with torch.no_grad():
            predicted_answers = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.bfloat16),
                max_new_tokens=max_new_tokens,)
        predicted_answers = tokenizer.decode(predicted_answers[:, :].cpu()[0], skip_special_tokens=True)
        
    elif 'idefics' in engine:
        prompts = [f"You are a helpful assistant.\n{task_instruction}\n"]
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                prompts.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
            prompts.append(f"\nUser: {n_shot_support[i]['question']}")
            #prompts.append("<end_of_utterance>")
            prompts.append(f"\nAssistant: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n")
        for query_image in query_images:
            prompts.append(query_image)
        prompts.append(f"\nUser: {query_text}")
        #prompts.append("<end_of_utterance>")
        prompts.append("\nAssistant:")
        inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to("cuda")
        exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

        generated_ids = model.generate(**inputs, 
                                       eos_token_id=exit_condition, 
                                       bad_words_ids=bad_words_ids, 
                                       max_new_tokens=max_new_tokens,
                                       do_sample=False)
        input_token_len = inputs['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(generated_ids[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    return predicted_answers