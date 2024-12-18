from setuptools import setup, find_packages

setup(
    name="viz_neutronics",
    py_modules=['viz_neutronics'],
    version="0.1",
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib'],  # Add dependencies here, e.g., ['numpy', 'pandas']
    description="A custom Python module",
    author="Shan Tan-Ya",
    author_email="st712@cam.ac.uk",
    url="https://github.com/Logan125/viz-neutronics",  # Optional
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)