import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
     name='physionet_challenge',  
     version='0.1',
     author="Kotzly",
     author_email="paullo.augusto@hotmail.com",
     description="Code for Physionet Challenge 2020.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/Kotzly/physionet-challenge-2020",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ],
     python_requires='>=3.8',
     install_requires=required,
 )
