import hydra

from diffusion_policy.workspace.base_workspace import BaseWorkspace

from nav2d import make, register


@hydra.main(version_base=None)
def main(_):
    # cfg = make("config.image_dummy#diffusion_policy#cnn")
    # cfg = make("config.image_dummy-binary#diffusion_policy#cnn")
    # cfg = make("config.image_dummy-a_star#diffusion_policy#cnn")
    cfg = make("config.image_dummy-a_star#diffusion_policy#cnn_24")
    # cfg = make("config.image_maze#diffusion_policy#cnn")

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    import nav2d.exp.diffusion_policy_vis_dataset

    main()

