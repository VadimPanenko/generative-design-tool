"""
что добавить (может быть...)
кэш для ранее загруженных моделей. сейчас каждый вызов скрипта загружает модель на gpu, выполняет генерацию и gpu оплностью очищается. 
почти все варианты генерации (basic, fill, depth, redux) требуют t5 и clip - зачем их загружать каждый раз заново?
однако, вариант redux включает загрузку feature_extractor, image_encoder, image_embedder - эти модули используются только с redux (и в нынешней реализации flux depth)
модули transformer, vae, scheduler для варианта генерации basic и fill отличаются
если и использовать кэш, то хотя при однообразном использовании варианта генерации basic переиспользуемые модули не будут загружаться, но:
при доработке результата (полученного при помощи basic) при помощи flux fill или flux depth будут оставаться неиспользуемые модули и забивать память.
как будто выигрыш в скорости генерации не сильно высок (модули загружаются в течении 5-15 секунд...)
"""




import torch
from diffusers import (
    FluxPipeline,
    FluxFillPipeline,
    FluxImg2ImgPipeline,
    FluxControlInpaintPipeline,
    FluxPriorReduxPipeline,
    FluxControlNetPipeline,
)
from utils import (
    load_models, get_depth,
    load_conditions,
    prepare_image_for_flux_upscale,
    get_prompt_and_image_embeds_scaled,
    remove_bg,
    load_qwen,
    vl_choose, # это могла бы быть функция, которая принимает изображения, отдаёт их VL модели, а та выбирает лучшее
)
import os
from pathlib import Path

def generate_image(
        mode,
        vl_cotrol,
        prompt,
        prompt_2,
        negative_prompt,
        negative_prompt_2,
        true_cfg_scale,
        height,
        width,
        upscale_strength,
        image_path,
        mask_path,
        reference_path,
        guidance_scale,
        num_images_per_prompt,
        num_inference_steps,
        controlnet_conditioning_scale,
        depth_strength,
        img2img_strength,
        text_t5_scale,
        text_clip_scale,
        image_siglip_scale,
        seed,
        output_path,
        device,
        aux_device,
        dtype
        ):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if mode in ["basic", "4bit"]:
        transformer, vae, scheduler, text_encoder, text_encoder_2, tokenizer, tokenizer_2 = load_models(mode, device, dtype)
        pipe = FluxPipeline(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            ).to(device)
        result = pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed)
            ).images
    
    elif mode in ["img2img", "img2img_4bit"]:
        if mode == "img2img":
            mode = "basic"
            transformer, vae, scheduler, text_encoder, text_encoder_2, tokenizer, tokenizer_2 = load_models(mode, device, dtype)
        else:
            mode = "basic_4bit"
            transformer, vae, scheduler, text_encoder, text_encoder_2, tokenizer, tokenizer_2 = load_models(mode, device, dtype)
        image, mask, _, _ = load_conditions(image_path, mask_path)
        image.resize((height, width))
        pipe = FluxImg2ImgPipeline(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            ).to(device)
        result = pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            true_cfg_scale=true_cfg_scale,
            image=image,
            height=height,
            width=width,
            strength=img2img_strength,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed)
            ).images
    
    elif mode in ["fill", "fill_4bit"]:
        transformer, vae, scheduler, text_encoder, text_encoder_2, tokenizer, tokenizer_2 = load_models(mode, device, dtype)
        image, mask, height, width = load_conditions(image_path, mask_path)
        pipe = FluxFillPipeline(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            ).to(device)
        result = pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            image=image,
            mask_image=mask,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed)
            ).images

    elif mode in ["depth", "depth_4bit"]:
        transformer, vae, scheduler, text_encoder, text_encoder_2, tokenizer, tokenizer_2, feature_extractor, image_encoder, image_embedder = load_models(mode, device, dtype)
        text_pipe = FluxPipeline(
            image_encoder=None,
            transformer=None,
            vae=None,
            scheduler=None,
            feature_extractor=None,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            ).to(device)
        img_pipe = FluxPriorReduxPipeline(
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            image_embedder=image_embedder,
            text_encoder=None,
            text_encoder_2=None,
            )
        pipe = FluxControlInpaintPipeline(
            transformer=transformer,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            scheduler=scheduler, 
            vae=vae,
            ).to(device)
        control_image = get_depth(image_path, device)
        image, mask, height, width = load_conditions(image_path, mask_path)
        prompt_embeds, pooled_prompt_embeds, _, _ = get_prompt_and_image_embeds_scaled(
            text_pipe=text_pipe,
            img_pipe=img_pipe,
            height=height,
            width=width,
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            ref_image=reference_path,
            num_images_per_prompt=num_images_per_prompt,
            text_t5_scale = text_t5_scale,
            text_clip_scale = text_clip_scale,
            image_siglip_scale = image_siglip_scale,
            device=device,
            )
        result = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            image=image,
            control_image=control_image,
            mask_image=mask,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            strength=depth_strength,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=torch.Generator("cpu").manual_seed(seed),
            ).images
        
    elif mode in ["redux", "redux_4bit"]:
        transformer, vae, scheduler, text_encoder, text_encoder_2, tokenizer, tokenizer_2, feature_extractor, image_encoder, image_embedder = load_models(mode, device, dtype)
        text_pipe = FluxPipeline(
            image_encoder=None,
            transformer=None,
            vae=None,
            scheduler=None,
            feature_extractor=None,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            ).to(device)
        img_pipe = FluxPriorReduxPipeline(
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            image_embedder=image_embedder,
            text_encoder=None,
            text_encoder_2=None,
            )
        pipe = FluxPipeline(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            ).to(device)
        prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds = get_prompt_and_image_embeds_scaled(
            text_pipe=text_pipe,
            img_pipe=img_pipe,
            height=height,
            width=width,
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            ref_image=image_path,
            num_images_per_prompt=num_images_per_prompt,
            text_t5_scale = text_t5_scale,
            text_clip_scale = text_clip_scale,
            image_siglip_scale = image_siglip_scale,
            device=device,
            )
        result = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=torch.Generator("cpu").manual_seed(seed),
            ).images

    elif mode in ["flux_upscale", "flux_upscale_4bit"]:
        controlnet, transformer, vae, scheduler = load_models(mode, device, dtype, upscale_strength)
        pipe = FluxControlNetPipeline(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            controlnet=controlnet,
            ).to(device)
        
        control_image = prepare_image_for_flux_upscale(image_path, upscale_strength)
        prompt_embeds = torch.zeros((1, 512, 4096), device=device, dtype=dtype)
        pooled_prompt_embeds = torch.zeros((1, 768), device=device, dtype=dtype)

        latents = pipe(
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale,
            height=control_image.size[1],
            width=control_image.size[0],
            output_type="latent"
            ).images

        latents = pipe._unpack_latents(latents, control_image.size[1], control_image.size[0], pipe.vae_scale_factor)
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

        if aux_device is not None:
            latents = latents.to(aux_device)
            pipe.vae = pipe.vae.to(aux_device)
        
        with torch.no_grad():
            image = pipe.vae.decode(latents).sample

        result = pipe.image_processor.postprocess(image, output_type="pil")

    elif mode == "remove_bg":
        model = load_models(mode, device, dtype)
        result = remove_bg(image=image_path, model=model, model_input_size=(1024, 1024), device=device)
        result.save(output_path)
        return
    
    if not isinstance(result, list):
        result = [result]
    
    if vl_cotrol and num_images_per_prompt > 1:
        model, processor = load_qwen(dtype, device=aux_device if aux_device else device)
        img = vl_choose(model=model, processor=processor, pil_images=result, user_prompt=prompt, device=aux_device if aux_device else device)
        img.save(output_path)
        # for i, image in enumerate(result):
        #     base_path, ext = os.path.splitext(output_path)
        #     current_output_path = f"{base_path}_{i}_pre_vl{ext}"
        #     image.save(current_output_path)
    else:
        for i, image in enumerate(result):
            base_path, ext = os.path.splitext(output_path)
            current_output_path = f"{base_path}_{i}{ext}"
            image.save(current_output_path)