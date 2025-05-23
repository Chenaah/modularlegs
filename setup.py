from setuptools import setup, find_packages


setup(
    name="modular_legs",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pyyaml",
        "omegaconf",
        "gymnasium",
        "sbx-rl",
        "wandb",
        "mujoco",
        "lxml",
        "pytorch_lightning",
        "imageio",
        "tensorboard",
        "moviepy",
        "torch==2.7.0",
        "jax[cuda12]",
        "ray",
        "mujoco-mjx",
        "gdown",
    ],
)