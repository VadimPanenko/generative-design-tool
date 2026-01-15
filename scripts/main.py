import argparse
from modes import generate_image
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Генерация изображения средствами FLUX")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "basic",
            "basic_4bit",
            "img2img",
            "fill",
            "fill_4bit",
            "depth", # нужно тогда загрузить и эту модель на сервер
            "depth_4bit",
            "redux",
            "redux_4bit",
            "flux_upscale",
            "flux_upscale_4bit",
            "remove_bg",
        ],
        help="Вариант генерации"
    )
    parser.add_argument("--vl_cotrol", type=bool, default=False, help="Использовать или не использовать qwen для отбора изображения") # сейчас это никак не реализовано
    parser.add_argument("--prompt", type=str, default="", help="Промпт для генерации; Если задан prompt_2, то prompt идёт в CLIP")
    parser.add_argument("--prompt_2", type=str, default=None, help="Промпт для генерации с учётом особенностей T5")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Негативный промпт; Если задан negative_prompt_2, то negative_prompt идёт в CLIP")
    parser.add_argument("--negative_prompt_2", type=str, default=None, help="Негативный промпт с учётом особенностей T5")
    parser.add_argument("--true_cfg_scale", type=float, default=1.0, help="")
    parser.add_argument("--height", type=int, default=1024, help="Высота изображения")
    parser.add_argument("--width", type=int, default=1024, help="Ширина изображения")
    parser.add_argument("--upscale_strength", type=int, default=2, help="Сила upscale; очень gpu intense!")
    parser.add_argument("--image", type=str, help="Путь к изменяемому изображению")
    parser.add_argument("--mask", type=str, help="Путь к маске")
    parser.add_argument("--reference", type=str, help="Путь к референсному изображению")
    parser.add_argument("--guidance_scale", type=float, help="Сила следования промпту; для разных модулей разная чувствительность к порядку значений этого параметра!")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Количество картинок в выдаче")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Количество шагов диффузии")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.6, help="")
    parser.add_argument("--depth_strength", type=float, default=1.0, help="")
    parser.add_argument("--img2img_strength", type=float, default=0.9, help="")
    parser.add_argument("--text_t5_scale", type=float, default=1.0, help="Степень влияния эмбеддингов текста, полученных при помощи T5")
    parser.add_argument("--text_clip_scale", type=float, default=1.0, help="Степень влияния эмбеддингов текста, полученных при помощи CLIP")
    parser.add_argument("--image_siglip_scale", type=float, default=1.0, help="Степень влияния эмбеддингов референсного изображения, полученных при помощи SIGLIP")
    parser.add_argument("--seed", type=int, default=0, help="Для воспроизводимости")
    parser.add_argument("--output", type=str, default="output.png", help="Результат")
    parser.add_argument("--device", type=str, default="cuda:2", help="Устройство, на котором будет проходить генерация")
    parser.add_argument("--aux_device", type=str, default=None, help="Доп. устройство, на котором будет проходить генерация")
    parser.add_argument("--dtype", type=torch.dtype, default=torch.bfloat16, help="Точность")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_image(
        mode=args.mode,
        vl_cotrol=args.vl_cotrol,
        prompt=args.prompt,
        prompt_2=args.prompt_2,
        negative_prompt=args.negative_prompt,
        negative_prompt_2=args.negative_prompt_2,
        true_cfg_scale=args.true_cfg_scale,
        height=args.height,
        width=args.width,
        upscale_strength=args.upscale_strength,
        image_path=args.image, 
        mask_path=args.mask,
        reference_path=args.reference,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        num_inference_steps=args.num_inference_steps,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        depth_strength=args.depth_strength,
        img2img_strength=args.img2img_strength,
        text_t5_scale=args.text_t5_scale,
        text_clip_scale=args.text_clip_scale,
        image_siglip_scale=args.image_siglip_scale,
        seed=args.seed,
        output_path=args.output,
        device=args.device,
        aux_device=args.aux_device,
        dtype=args.dtype,
    )