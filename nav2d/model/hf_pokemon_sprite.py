"""
https://huggingface.co/sWizad/pokemon-trainer-sprite-pixelart
"""
from diffusers import AutoPipelineForText2Image
import torch

from nav2d import register, make


@register("pipeline.text2img.pokemon-trainer-sprite-pixelart")
def pipeline(device='cuda'):
    pipeline = AutoPipelineForText2Image.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16).to(device)
    pipeline.load_lora_weights('sWizad/pokemon-trainer-sprite-pixelart', weight_name='pk_trainer_xl_v1.safetensors')

    # TODO: extend pipeline object
    # def _pipeline(prompt, *args, **kwargs):
    #     images = pipeline(prompt, *args, **kwargs).images
    #     print(len(images))
    #     image = images[0]
    #     # scale down to 96 x 96
    #     size = 96, 96
    #     image = image.resize(size)
    #     # crop
    #     border = 10
    #     image = image.crop((border, border, size[0] - border, size[1] - border))
    #     return image

    def _pipeline(prompts, *args, **kwargs):
        # Process multiple prompts at once
        result_images = pipeline(prompts, *args, **kwargs).images
        
        processed_images = []
        for image in result_images:
            # Scale down to 96 x 96
            size = 96, 96
            image = image.resize(size)
            # Crop
            border = 10
            image = image.crop((border, border, size[0] - border, size[1] - border))
            processed_images.append(image)
        
        return processed_images

    return _pipeline


if __name__ == "__main__":
    # pipe = make("pipeline.text2img.pokemon-trainer-sprite-pixelart")
    # image = pipe('close-up, 1girl, solo, hood, simple background', )
    # # image = pipe('snowman, simple background').images[0]
    # # image = pipe('a ranger with a bow, simple background').images[0]
    # # image = pipe('a treasure chest, simple background').images[0]
    # image = pipe('a wall')
    # image.save('pokemon_trainer.png')

    pipe = make("pipeline.text2img.pokemon-trainer-sprite-pixelart", device="cuda:1")
    prompts = ['close-up, 1girl, solo, hood, simple background', 'wall texture', 'a ranger with a bow, simple background']
    images = pipe(prompts)

    for i, image in enumerate(images):
        image.save(f'pokemon_trainer_{i}.png')