import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('brainmontage/_version.py') as f:
    version=f.readline().split("=")[-1].split()[-1].replace("'","")

setuptools.setup(
    name="brainmontage", # Replace with your own username
    version=version,
    author="Keith Jamison",
    author_email="keith.jamison@gmail.com",
    description="Generate brain ROI figures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kjamison/brainmontageplot",
    packages=setuptools.find_packages(),
    install_requires = required,
    package_data={'brainmontage':['atlases/*','lookups/*']},
    include_package_data = True,
    entry_points = {'console_scripts':['brainmontage=brainmontage.brainmontage:run_montageplot']},
    classifiers= [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5,<3.11',
)
