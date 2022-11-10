import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as req_file:
    reqs = [l.strip() for l in req_file if l.strip() and not l.strip().startswith("#")]

setuptools.setup(
    name="deepface-trcp",
    version="0.0.120",
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
    install_requires=reqs
)
