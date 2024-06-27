from pathlib import Path
from nav2d.env.texture import load_texture_set
import torch
from PIL import Image
import open_clip


save_dir = Path(__file__).parent / "figures" / "texture_clip"
texture_dir = Path(__file__).parent / ".." / "output" / "texture" / "textures_subthemes"


if __name__ == "__main__":    
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir="/storage/nacloos/.hf")
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # image = Image.open("docs/CLIP.png")
    # texture_path = texture_dir / "Medieval Castle" / "subtheme0" / "run2"
    texture_path = texture_dir / "Medieval Castle" / "subtheme1" / "run2"
    texture_set = load_texture_set(texture_path, 64)
    image = texture_set["goal"]

    image = preprocess(image).unsqueeze(0)
    text = tokenizer(["agent", "object", "wall", "empty"])

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

