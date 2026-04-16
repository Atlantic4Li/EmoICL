"""
CIRCLES integration for IntentICL.

Implements attribute-guided composed image retrieval (CIRCLES) adapted for
IntentICL's data format and model loading pipeline.

Reference: Xiong et al., "Retrieving Counterfactuals Improves Visual
In-Context Learning", CVPR 2026.
"""

import os
import copy
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy-loaded CLIP model (shared across calls within a session)
# ---------------------------------------------------------------------------
_clip_model = None
_clip_processor = None
_clip_model_name = None

# Cache for support-set text features (keyed by id(support_meta))
_support_text_features_cache: Dict[int, Dict[str, torch.Tensor]] = {}


CIRCLES_INSTRUCTION = (
    "You are an image description expert. You are given an original image and "
    "manipulation text. Your goal is to generate a target image description "
    "that reflects the changes described based on manipulation intents while "
    "retaining as much image content from the original image as possible.\n\n"
    "## Guidelines on generating Target Image Description\n"
    "- The target image description should be complete and cover various semantic aspects.\n"
    "- The target image description only contains the target image content and "
    "needs to be as simple as possible. Minimize aesthetic descriptions.\n\n"
    "## Response format\n"
    '{"Target Image Description": <target_image_description>}'
)


# ========================================================================== #
#                            CLIP helpers                                     #
# ========================================================================== #

def _ensure_clip(clip_model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cuda"):
    """Lazily load a HuggingFace CLIPModel + CLIPProcessor."""
    global _clip_model, _clip_processor, _clip_model_name
    if _clip_model is not None and _clip_model_name == clip_model_name:
        return _clip_model, _clip_processor

    from transformers import CLIPModel, CLIPProcessor as _CLIPProc
    _clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    _clip_processor = _CLIPProc.from_pretrained(clip_model_name)
    _clip_model_name = clip_model_name
    return _clip_model, _clip_processor


def _get_text_features(texts: List[str],
                       clip_model_name: str = "openai/clip-vit-base-patch32",
                       device: str = "cuda") -> torch.Tensor:
    """Return L2-normalised CLIP text features (N, D)."""
    model, processor = _ensure_clip(clip_model_name, device)
    inputs = processor(
        text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return F.normalize(features, p=2, dim=-1)


def _get_support_text_features(
    support_meta: List[Dict],
    clip_model_name: str,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Compute (and cache) CLIP text features for support-set answers."""
    cache_key = id(support_meta)
    if cache_key in _support_text_features_cache:
        return _support_text_features_cache[cache_key]

    ids: List[str] = []
    texts: List[str] = []
    for item in support_meta:
        ids.append(str(item["img_id"]))
        texts.append(str(item.get("answer", item.get("question", ""))))

    batch_size = 128
    all_feats: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        feats = _get_text_features(texts[i:i + batch_size], clip_model_name, device)
        all_feats.append(feats.cpu())
    all_feats_cat = torch.cat(all_feats, dim=0)

    features_dict = {img_id: all_feats_cat[j] for j, img_id in enumerate(ids)}
    _support_text_features_cache[cache_key] = features_dict
    return features_dict


# ========================================================================== #
#                      Generic single-turn VLM generation                     #
# ========================================================================== #

def vlm_single_generate(
    engine: str,
    model: Any,
    tokenizer: Any,
    processor: Any,
    image_paths: List[str],
    text_prompt: str,
    data_path: str,
    max_new_tokens: int = 512,
) -> str:
    """Run a single-turn VLM generation with one or more images + text.

    Supports the same engines as IntentICL's ``model_inference.ICL_inference``.
    """

    if "qwen-vl" in engine:
        inputs_list: list = []
        for img_path in image_paths:
            inputs_list.append({"image": os.path.join(data_path, img_path)})
        inputs_list.append({"text": f"You are a helpful assistant.\nUser: {text_prompt}\nAssistant:"})
        total_inputs = tokenizer.from_list_format(inputs_list)
        inputs = tokenizer(total_inputs, return_tensors="pt").to(model.device)
        with torch.no_grad():
            pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
        input_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(pred[:, input_len:].cpu()[0], skip_special_tokens=True)

    if "llava" in engine:
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

        images = []
        input_text = ""
        for img_path in image_paths:
            images.append(Image.open(os.path.join(data_path, img_path)).convert("RGB"))
            input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
        input_text += text_prompt

        image_tensor = torch.stack([
            processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
            for img in images
        ]).half().cuda()

        conv_mode = "qwen_1_5" if "onevision" in engine else "llava_v1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        with torch.inference_mode(), torch.cuda.amp.autocast():
            gen_ids = model.generate(
                input_ids, images=image_tensor,
                do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1,
            )
        return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

    if "idefics" in engine:
        prompts: list = []
        for img_path in image_paths:
            prompts.append(Image.open(os.path.join(data_path, img_path)).convert("RGB"))
        prompts.append(f"\nUser: {text_prompt}")
        prompts.append("\nAssistant:")

        inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to("cuda")
        exit_cond = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_ids = processor.tokenizer(
            ["<image>", "<fake_token_around_image>"], add_special_tokens=False
        ).input_ids
        gen_ids = model.generate(
            **inputs, eos_token_id=exit_cond, bad_words_ids=bad_ids,
            max_new_tokens=max_new_tokens, do_sample=False,
        )
        input_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(gen_ids[:, input_len:].cpu()[0], skip_special_tokens=True)

    if "internlm-x" in engine:
        images = []
        input_text = ""
        for img_path in image_paths:
            img = Image.open(os.path.join(data_path, img_path)).convert("RGB")
            images.append(model.vis_processor(img))
            input_text += "<ImageHere>"
        input_text += text_prompt
        image_t = torch.stack(images).to(torch.bfloat16).cuda()
        out, _ = model.chat(
            tokenizer, query=input_text, image=image_t,
            history=[], do_sample=False, max_new_tokens=max_new_tokens,
        )
        return out

    if "flamingo" in engine or "openflamingo" in engine:
        images = []
        input_text = ""
        for img_path in image_paths:
            images.append(Image.open(os.path.join(data_path, img_path)).convert("RGB"))
            input_text += "<image>"
        input_text += text_prompt
        vision_x = torch.cat([processor(img).unsqueeze(0) for img in images], dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        lang_x = tokenizer([input_text], return_tensors="pt")
        with torch.no_grad():
            pred = model.generate(
                vision_x=vision_x.to(torch.bfloat16).cuda(),
                lang_x=lang_x["input_ids"].cuda(),
                attention_mask=lang_x["attention_mask"].cuda(),
                max_new_tokens=max_new_tokens, do_sample=False,
            )
        input_len = lang_x["input_ids"].shape[1]
        return tokenizer.decode(pred[:, input_len:].cpu()[0], skip_special_tokens=True)

    if "otter" in engine:
        images = []
        input_text = ""
        for img_path in image_paths:
            images.append(Image.open(os.path.join(data_path, img_path)).convert("RGB"))
            input_text += "<image>"
        input_text += f"User: {text_prompt}\nGPT:<answer>"
        vision_x = processor.preprocess(images, return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        lang_x = model.text_tokenizer([input_text], return_tensors="pt")
        bad_ids = tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
        with torch.no_grad():
            pred = model.generate(
                vision_x=vision_x.to(model.device),
                lang_x=lang_x["input_ids"].to(model.device),
                attention_mask=lang_x["attention_mask"].to(model.device),
                max_new_tokens=max_new_tokens, do_sample=False,
                bad_words_ids=bad_ids,
            )
        input_len = lang_x["input_ids"].shape[1]
        return tokenizer.decode(pred[:, input_len:].cpu()[0], skip_special_tokens=True)

    if "emu2-chat" in engine:
        images = []
        input_text = ""
        for img_path in image_paths:
            images.append(Image.open(os.path.join(data_path, img_path)).convert("RGB"))
            input_text += "[<IMG_PLH>]"
        input_text += f"[{text_prompt}"
        inputs = model.build_input_ids(text=[input_text], tokenizer=tokenizer, image=images)
        with torch.no_grad():
            pred = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.bfloat16),
                max_new_tokens=max_new_tokens,
            )
        return tokenizer.decode(pred[:, :].cpu()[0], skip_special_tokens=True)

    raise NotImplementedError(f"Engine '{engine}' not supported for CIRCLES generation")


# ========================================================================== #
#                   CIRCLES attribute & caption generation                     #
# ========================================================================== #

def identify_attributes(
    engine: str,
    model: Any,
    tokenizer: Any,
    processor: Any,
    image_paths: List[str],
    question: str,
    data_path: str,
    num_attributes: int = 1,
    max_new_tokens: int = 512,
) -> List[str]:
    """Use the VLM to identify key visual attributes of the query image."""
    prompt = (
        "Identify the key attributes of the following image that are most "
        "relevant to answering the question.\n"
        f"Question: {question}\n"
        f"Please list the top {num_attributes} key attributes as short phrases "
        "in a section named '### Attributes', one per line, ordered from most "
        "to least important."
    )
    response = vlm_single_generate(
        engine, model, tokenizer, processor,
        image_paths, prompt, data_path, max_new_tokens,
    )

    # Parse structured attribute list from VLM output
    text = response.split("### Attributes")[-1].strip() if "### Attributes" in response else response
    lines = text.split("\n")
    attributes: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        # Strip common list prefixes
        if line.startswith(("- ", "* ")):
            line = line[2:].strip()
        elif line.startswith(("• ",)):
            line = line[2:].strip()
        # Strip leading numbering like "1." or "1)"
        if line and line[0].isdigit():
            for idx, ch in enumerate(line):
                if ch in ".):":
                    if line[:idx].isdigit():
                        line = line[idx + 1:].strip()
                        break
        if line:
            attributes.append(line)
        if len(attributes) >= num_attributes:
            break
    return attributes


def generate_modified_caption(
    engine: str,
    model: Any,
    tokenizer: Any,
    processor: Any,
    image_paths: List[str],
    attribute: str,
    data_path: str,
    max_new_tokens: int = 512,
) -> str:
    """Use the VLM to produce a counterfactual caption by modifying *attribute*."""
    prompt = (
        f"{CIRCLES_INSTRUCTION}\n"
        f"Manipulation Text: Change the attribute '{attribute}' to a different "
        "plausible value. Ensure the modified caption is concise and contains "
        "no more than 77 tokens."
    )
    response = vlm_single_generate(
        engine, model, tokenizer, processor,
        image_paths, prompt, data_path, max_new_tokens,
    )
    # Try to extract the structured answer
    if "Target Image Description" in response:
        return response.split("Target Image Description")[-1].strip('":} \n')
    return response.strip()


# ========================================================================== #
#                        Main CIRCLES retrieval                               #
# ========================================================================== #

def retrieve_circles_demos(
    query: Dict[str, Any],
    support_meta: List[Dict[str, Any]],
    n_shot: int,
    similarity_data: Dict,
    support_features: Dict[str, torch.Tensor],
    model: Any,
    tokenizer: Any,
    processor: Any,
    engine: str,
    data_path: str,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    num_attributes: int = 1,
    attribute_k: Optional[int] = None,
    attributes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """CIRCLES retrieval: standard RICES + attribute-guided composed retrieval.

    Returns
    -------
    dict with keys:
        ``original_retrievals``  – list of support items (standard RICES)
        ``composed_retrievals``  – list of dicts, each containing
            ``attribute``, ``modified_caption``, ``retrieved_items``
    """
    if attribute_k is None:
        attribute_k = n_shot

    device = "cuda" if torch.cuda.is_available() else "cpu"
    support_dict = {str(item["img_id"]): item for item in support_meta}

    # ------------------------------------------------------------------
    # Stage 1: Original retrievals (RICES – image-image similarity)
    # ------------------------------------------------------------------
    query_id = str(query["img_id"])
    all_sims = {
        k: v for k, v in similarity_data.get(query_id, {}).items()
        if k in support_dict
    }
    sorted_sims = sorted(all_sims.items(), key=lambda x: x[1], reverse=True)
    original_retrievals = [
        copy.deepcopy(support_dict[s_id]) for s_id, _ in sorted_sims[:n_shot]
    ]

    # ------------------------------------------------------------------
    # Stage 2: Identify attributes (use VLM)
    # ------------------------------------------------------------------
    if attributes:
        attr_list = [a.strip() for a in attributes if isinstance(a, str) and a.strip()][:num_attributes]
    else:
        attr_list = identify_attributes(
            engine, model, tokenizer, processor,
            query.get("image", []),
            query.get("question", ""),
            data_path,
            num_attributes=num_attributes,
        )

    # ------------------------------------------------------------------
    # Stage 3: Composed retrieval for each attribute
    # ------------------------------------------------------------------
    # Build support image-feature matrix (ordered)
    support_ids_ordered = [str(item["img_id"]) for item in support_meta]
    support_feat_list = []
    valid_mask = []
    for sid in support_ids_ordered:
        if sid in support_features:
            support_feat_list.append(support_features[sid])
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    if not support_feat_list:
        # Fallback: no features available
        return {
            "original_retrievals": original_retrievals,
            "composed_retrievals": [],
        }

    support_feat_matrix = torch.stack(support_feat_list).float().to(device)
    # Normalise in case raw features aren't unit-length
    support_feat_matrix = F.normalize(support_feat_matrix, p=2, dim=-1)
    # Keep only the IDs that have features
    valid_support_ids = [sid for sid, ok in zip(support_ids_ordered, valid_mask) if ok]

    composed_retrievals: List[Dict[str, Any]] = []
    for attr in attr_list:
        caption = generate_modified_caption(
            engine, model, tokenizer, processor,
            query.get("image", []), attr, data_path,
        )
        if not caption:
            composed_retrievals.append({
                "attribute": attr,
                "modified_caption": "",
                "retrieved_items": [],
            })
            continue

        # Compute CLIP text features for the modified caption
        caption_features = _get_text_features([caption], clip_model_name, device)

        # Composed similarity: caption ↔ support images
        sim = (caption_features @ support_feat_matrix.T).squeeze(0)

        k = min(attribute_k, len(valid_support_ids))
        if k == 0:
            composed_retrievals.append({
                "attribute": attr,
                "modified_caption": caption,
                "retrieved_items": [],
            })
            continue

        top_indices = torch.topk(sim, k).indices.cpu().tolist()
        items = [copy.deepcopy(support_dict[valid_support_ids[i]]) for i in top_indices]
        composed_retrievals.append({
            "attribute": attr,
            "modified_caption": caption,
            "retrieved_items": items,
        })

    return {
        "original_retrievals": original_retrievals,
        "composed_retrievals": composed_retrievals,
    }
