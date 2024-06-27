"""
https://huggingface.co/nerijs/pixel-art-xl

pip install peft
"""
import torch
from diffusers import DiffusionPipeline

from nav2d import register, make


@register("pipeline.text2img.pixel_art_xl")
def pixel_art_xl_pipeline():
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DiffusionPipeline.from_pretrained(base_model_id, variant="fp16", torch_dtype=torch.float16).to("cuda")
    pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors")
    
    def _pipe(prompt):
        image = pipe(prompt).images[0]
        return image
    return _pipe


if __name__ == "__main__":
    pipe = make("pipeline.text2img.pixel_art_xl")
    
    prompt = "Goal (Destination): Description: The goal is marked by a flag on a pole, fluttering in the arctic wind. The flag could be a bright color like red or yellow, providing a stark contrast to the icy surroundings. Below the flag, a snow mound indicates the checkpoint base. Style: 2D, pixel art, vibrant against a predominantly white and blue background."
    prompt = "Agent (Player Character): Description: The agent is an explorer dressed in a heavy, insulated parka designed for arctic conditions. The outfit is white and gray to blend with the snowy environment, complete with fur-lined hood and goggles. The character is shown in a standing pose, ready for exploration. Style: 2D, pixel art, detailed enough to show clothing texture and cold weather gear."
    # image = pipe( "pixel art, modern computer, simple").images[0]
    image = pipe(prompt)
    image.save("computer.png")



