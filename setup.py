from setuptools import setup, find_packages


setup(
    name='nav2d',
    version="0.0.1",
    packages=[package for package in find_packages() if package.startswith('nav2d')],
    install_requires=[
        "gym==0.26.2",
        "pygame==2.1.2",
        "shapely==1.8.4",
        "pymunk==6.2.1",
        "opencv-python",
        "scikit-image==0.19.3",
        "torch",
        "gdown",
        "zarr",
        "matplotlib",
        "einops",
        "minari",
        "gymnasium-robotics",
        "scikit-video",
        "hydra-core==1.2.0",
        # "robomimic==0.2.0",
        "av==10.0.0",
    ],
    description='',
    author=''
)
