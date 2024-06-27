from collections import defaultdict
from pathlib import Path

from matplotlib import pyplot as plt
from nav2d.env.texture import load_texture_set
from dreamsim import dreamsim
from PIL import Image
from nav2d.exp.straight_lines_train import generate_straight_lines_texture_data
import numpy as np
import pandas as pd
import torch


texture_dir = Path(__file__).parent / ".." / "output" / "texture" / "textures_subthemes"
save_dir = Path(__file__).parent / "figures"/ "straight_lines_texture"
save_dir.mkdir(parents=True, exist_ok=True)


def visualize_textures(obj_type, save_dir):
    texture_size = 45
    textures = defaultdict(list)

    theme = "*"
    theme = "Medieval Castle"
    for p in texture_dir.glob(f"{theme}/*/*"):
        print(p)
        theme = p.parent.parent.stem
        subtheme = p.parent.stem
        run = p.stem

        texture_set = load_texture_set(p, texture_size)

        for k, img in texture_set.items():
            textures["theme"].append(theme)
            textures["subtheme"].append(subtheme)
            textures["run"].append(run)
            textures["image"].append(img)
            textures["object_type"].append(k)

    textures = pd.DataFrame(textures)

    # select agent images
    agent_images = textures[textures["object_type"] == obj_type]["image"]
    save_dir /= obj_type
    save_dir.mkdir(parents=True, exist_ok=True)

    agent_images = agent_images.tolist()
    print(agent_images)
    model, preprocess = dreamsim(pretrained=True, cache_dir="/storage/nacloos/.hf", device="cpu") 


    images = [preprocess(img)[0] for img in agent_images]
    images = torch.stack(images)
    print(images.shape)

    embedding = model.embed(images)
    print(embedding.shape)

    from sklearn.decomposition import PCA

    pca = PCA()
    pca.fit(embedding.detach().numpy())
    plt.figure()
    plt.plot(pca.explained_variance_ratio_)
    plt.savefig(save_dir / "explained_variance_ratio.png")

    # plot 2D projection
    pca = PCA(n_components=2)
    pca.fit(embedding.detach().numpy())
    embedding_2d = pca.transform(embedding.detach().numpy())

    plt.figure()
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], marker=".")
    plt.savefig(save_dir / "2d_projection.png")


    # plot small image at the corresponding position
    plt.figure(dpi=500)
    w = (embedding_2d[:, 0].max() - embedding_2d[:, 0].min()) / 30
    h = (embedding_2d[:, 1].max() - embedding_2d[:, 1].min()) / 30

    for i, (x, y) in enumerate(embedding_2d):
        img = agent_images[i]
        # plot img at position x, y
        plt.imshow(img, extent=(x, x+w, y, y+h))
    plt.xlim(embedding_2d[:, 0].min()-w, embedding_2d[:, 0].max()+w)
    plt.ylim(embedding_2d[:, 1].min()+h, embedding_2d[:, 1].max()+h)
    plt.savefig(save_dir / "2d_projection_images.png")


    # tSNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    embedding_2d = tsne.fit_transform(embedding.detach().numpy())

    plt.figure()
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], marker=".")
    plt.axis('equal')
    plt.savefig(save_dir / "2d_projection_tsne.png")

    # plot small image at the corresponding position
    plt.figure(dpi=500)
    w = (embedding_2d[:, 0].max() - embedding_2d[:, 0].min()) / 30
    h = (embedding_2d[:, 1].max() - embedding_2d[:, 1].min()) / 30
    for i, (x, y) in enumerate(embedding_2d):
        img = agent_images[i]
        print(x, y)
        # plot img at position x, y
        plt.imshow(img, extent=(x, x+w, y, y+h))

    plt.xlim(embedding_2d[:, 0].min()-w, embedding_2d[:, 0].max()+w)
    plt.ylim(embedding_2d[:, 1].min()+h, embedding_2d[:, 1].max()+h)
    plt.axis('equal')
    plt.xlabel("tSNE 1")
    plt.ylabel("tSNE 2")
    plt.title(obj_type)
    plt.tight_layout()
    plt.savefig(save_dir / "2d_projection_images_tsne.png")



def plot_data(X, y, n_samples=10, save_dir=None):
    for i in range(n_samples):
        plt.figure()
        plt.imshow(X[i])

        colormap = plt.cm.plasma
        colors = np.linspace(0, 1, y[i].shape[0])
        plt.scatter(y[i][:, 1], y[i][:, 0], c=colors, cmap=colormap, marker=".")
        plt.axis('off')
        plt.axis('equal')
        plt.tight_layout()
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f"sample_{i}.png")


if __name__ == "__main__":    
    # obj_type = "agent"
    # obj_type = "goal"
    # visualize_textures(obj_type, save_dir)

    n_samples = 10
    grid_size = 96
    texture_size = 30
    n_steps = 64
    X, y = generate_straight_lines_texture_data(n_samples, grid_size, texture_size, n_steps=n_steps)
    plot_data(X, y, n_samples, save_dir / "data")


