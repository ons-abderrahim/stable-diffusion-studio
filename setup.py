from setuptools import setup, find_packages

setup(
    name="stable-diffusion-studio",
    version="1.0.0",
    description="Image generation with Stable Diffusion — inference, prompt engineering, and fine-tuning.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/yourusername/stable-diffusion-studio",
    license="MIT",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "diffusers>=0.28.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "safetensors>=0.4.3",
        "peft>=0.11.0",
        "Pillow>=10.3.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "gradio>=4.36.0",
        "PyYAML>=6.0.1",
    ],
    extras_require={
        "eval": ["torchmetrics[image]>=1.4.0", "open-clip-torch>=2.24.0"],
        "dev": [
            "pytest>=8.2.0",
            "black>=24.4.0",
            "isort>=5.13.0",
            "mypy>=1.10.0",
            "ruff>=0.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "sd-generate=scripts.generate:main",
            "sd-train-dreambooth=scripts.train_dreambooth:main",
            "sd-train-lora=scripts.train_lora:main",
            "sd-evaluate=scripts.evaluate:main",
        ]
    },
)
