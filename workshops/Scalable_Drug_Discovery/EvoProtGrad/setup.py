from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

with open('README.md', 'r') as f:
    readme = f.read()

setup(name='evo_prot_grad',
      version='0.2.1',
      description='Directed evolution of proteins with fast gradient-based discrete MCMC.',
      author='Patrick Emami',
      author_email='Patrick.Emami@nrel.gov',
      url='https://github.nrel.gov/NREL/EvoProtGrad/',
      python_requires='>=3.8',
      install_requires=requirements,
      long_description=readme,
      long_description_content_type='text/markdown',
      packages=find_packages(include=['evo_prot_grad',
                                      'evo_prot_grad.common',
                                      'evo_prot_grad.experts',
                                      'evo_prot_grad.models'],
                             exclude=['test']),
      license='BSD 3-Clause',
      keywords=['protein engineering', 'directed evolution', 'huggingface', 'protein language models', 'mcmc'],
      classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Natural Language :: English",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ]      
)
