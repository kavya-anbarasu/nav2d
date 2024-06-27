"""
Code adapted from https://huggingface.co/stabilityai/stable-cascade
"""
from pathlib import Path

import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline


def generate_image(prompt, negative_prompt="", save_path=None):
    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16)
    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16)

    print("Prompt:", prompt)
    if negative_prompt != "":
        print("Negative prompt:", negative_prompt)
    prior.enable_model_cpu_offload()
    prior_output = prior(
        prompt=prompt,
        height=1024,
        width=1024,
        negative_prompt=negative_prompt,
        guidance_scale=4.0,
        num_images_per_prompt=1,
        num_inference_steps=20
    )

    decoder.enable_model_cpu_offload()
    decoder_output = decoder(
        image_embeddings=prior_output.image_embeddings.to(torch.float16),
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        output_type="pil",
        num_inference_steps=10
    ).images[0]
    if save_path is not None:
        decoder_output.save(save_path)


if __name__ == "__main__":
    # prompts = [
    #     # "wall texture, 2d video game, gridworld, roguelike, simple, square",
    #     # "16-bit color depth pixel image for the character of a game, roguelike",
    #     # "background texture for empty space in 2d video game, gridworld, simple, square, roguelike",
    #     # "texture for the winning goal object in 2d video game, gridworld, simple, roguelike",
    #     "wall texture, cartoon style, 2d video game",
    #     "a small green blob with a hat, 2d video game",
    #     "background texture for empty space, cartoon style, 2d video game",
    #     "winning goal object sprite, cartoon style, 2d video game"
    # ]
    # prompts = [
    #     "Create a 2D pixel art texture of a game agent character designed for a gridworld game. The character should look like a futuristic robot, small and compact, with a smooth metallic body colored in shades of blue and silver. The robot should have glowing green eyes and articulated limbs, giving it a nimble appearance. The texture should be suitable for a 16x16 pixel grid.",
    #     "Design a 2D pixel art texture for a wall in a gridworld game. The wall should have a stone texture, appearing rugged and solid, with variations in gray tones to suggest depth and texture. Include slight mossy green accents to give a sense of age and weathering. The design should be repetitive without obvious tiling issues, suitable for a 16x16 pixel grid.",
    #     "Generate a 2D pixel art texture representing a goal location in a gridworld game. The goal should be depicted as a glowing portal, with swirling colors of blue and purple, giving it an ethereal, magical look. The portal should be encircled by a thin, golden frame, enhancing its significance. The texture should fit a 16x16 pixel grid.",
    #     "Create a 2D pixel art texture for an empty cell in a gridworld game. The texture should represent simple, flat ground with a subtle pattern of light brown and beige tiles, suggesting a dusty, sandy surface. The design should be minimal and not distract from game elements, suitable for a 16x16 pixel grid."
    # ]

    # prompts = [
    #     "A futuristic agent with glowing cybernetic enhancements, wearing a sleek, streamlined suit with neon accents that pulse softly. The character should appear agile and tech-savvy, with a visor displaying digital data. 2D pixel art style, vibrant neon colors predominantly in blues and purples.",
    #     "A wall made of digital panels that intermittently flicker with holographic data and advertisements. The wall should have a high-tech appearance, featuring glowing lines and cybernetic patterns that give a sense of a barrier that's part electronic and part physical. 2D pixel art, mix of dark tones and bright neon lights, such as electric blues and pinks.",
    #     "The goal is a brightly illuminated portal with swirling neon lights and a holographic archway. It should look inviting and high-tech, with streaming data patterns and a pulsating light that indicates the exit or next level. 2D pixel art, focusing on luminous and dynamic lighting effects in colors like cyan and magenta.",
    #     "The floor of the gridworld is a sleek, metallic surface with subtle neon tracings that grid the pathways. It should reflect a dim, ambient light and occasionally display soft digital effects or faint data streams. 2D pixel art, with a cooler palette of greys and muted neon accents to keep it understated yet futuristic."
    # ]

    prompts = {
        "agent": "Close-up of a futuristic agent, upper torso and head, featuring cybernetic enhancements and neon accents. Agent's suit is streamlined with contrasting highlights for visibility. Style: 2D pixel art, blues and purples.",
        "wall": "Digital wall with large, clear cybernetic patterns and bold glowing lines. Designed for seamless tiling and reduced detail for better downscaling. Style: 2D pixel art, dark tones with bright neon like blues and pinks.",
        "goal": "Simplified portal with large, dynamic neon swirls and holographic archway. High-contrast, minimal design to draw attention at smaller scales. Style: 2D pixel art, cyan and magenta.",
        "empty": "Floor with broad, muted neon tracings creating a grid pattern. Minimal digital effects, designed for seamless tiling and clarity at smaller sizes. Style: 2D pixel art, greys with muted neon accents."
    }
    prompt = list(prompts.values())

    save_names = [
        "agent.png",
        "wall.png",
        "goal.png",
        "empty.png",
    ]

    # TODO: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
    # prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16)
    # decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16)

    save_dir = Path(__file__).parent / "data" / "set8"
    save_dir.mkdir(exist_ok=True, parents=True)

    for i, prompt in enumerate(prompts):
        save_path = save_dir / save_names[i]
        negative_prompt = ""
        generate_image(prompt, negative_prompt, save_path)
        # save prompt and negative prompt to json
        save_path = save_dir / f"prompt{i}.json"
        with open(save_path, "w") as f:
            f.write(f'{{"prompt": "{prompt}", "negative_prompt": "{negative_prompt}"}}')
        print(f"Saved image to {save_path}")
