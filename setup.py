import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jtorch",
    version="0.0.6",
    author="jtorch",
    author_email="jtorch@qq.com",
    description="jtorch project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JITTorch/jtorch",
    project_urls={
        "Bug Tracker": "https://github.com/JITTorch/jtorch/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=["jtorch", "torch"],
    package_dir={"": "python"},
    package_data={'': ['*', '*/*', '*/*/*','*/*/*/*','*/*/*/*/*','*/*/*/*/*/*']},
    python_requires=">=3.7",
    install_requires=[
        "jittor>=1.3.5.40",
    ],
)