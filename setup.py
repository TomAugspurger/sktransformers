from setuptools import setup, find_packages
from os import path

import versioneer


here = path.abspath(path.dirname(__file__))

# with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
#     long_description = f.read()

setup(
    name='sktransformers',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),

    description='SKTransformers',
    long_description='SKTransformers',

    url='https://github.com/tomaugspurger/sktransformers',

    author='Tom Augspurger',
    author_email='tom.augspurger88@gmail.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # keywords='sample setuptools development',
    packages=find_packages(exclude=['docs', 'tests']),

    install_requires=['pandas', 'dask', 'scikit-learn', ],

    extras_require={
        'dev': [],
    },
)
