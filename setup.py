import pathlib

import setuptools

HERE = pathlib.Path(__file__).parent.absolute()

requirements = list()
with open(HERE / "requirements.txt") as f:
    for line in f:
        line = line.strip()
        if not line.startswith("#"):
            requirements.append(line)

setuptools.setup(
    name="pytorch_faster_rcnn_module",
    version="0.0.1",
    author="Hamka Satria",
    author_email="muhhamkasatrianto@gmail.com",
    url="https://github.com/hamkasatria/PyTorch-Object-Detection-Faster-RCNN-Tutorial",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
)
