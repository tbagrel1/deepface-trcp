import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepface-trcp",
    version="0.0.91",
    author="Sefik Ilkin Serengil, Thomas Bagrel",
    author_email="serengil@gmail.com",
    description="A Lightweight Face Recognition and Facial Attribute Analysis Framework (Age, Gender, Emotion, Race) for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tbagrel1/deepface-trcp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.5.5',
    install_requires=["numpy>=1.14.0", "pandas>=0.23.4", "tqdm>=4.30.0", "gdown>=3.10.1", "Pillow>=5.2.0", "opencv-python>=4.5.5.64", "tensorflow>=1.9.0", "keras>=2.2.0", "Flask>=1.1.2", "mtcnn>=0.1.0", "retina-face>=0.0.1", "fire>=0.4.0"]
)
