from setuptools import setup, find_packages
import re
import io

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open('nixtract/__init__.py', encoding='utf_8_sig').read()
    ).group(1)

test_deps = ['pytest-cov',
             'pytest']

extras = {
    'test': test_deps,
}

setup(
    name='nixtract',
    version=__version__,
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*",
                                    "tests"]),
    license='MIT',
    author='Dan Gale',
    maintainer_email="d.gale@queensu.ca",
    description=("A unified interface for timeseries extraction from different "
                 "functional neuroimaging file types"),
    long_description=open('README.md').read(),
    url='https://github.com/danjgale/nixtract',
    python_requires='>=3.6.0',
    install_requires=[
        'numpy>=1.16.5',
        'pandas>=1.1.0',
        'nibabel>=3.2.0',
        'nilearn>=0.7.1',
        'natsort>=7.1.1',
        'scipy>=1.5.0',
        'scikit-learn>=0.24.1',
        'load_confounds'
    ],
    tests_require=test_deps,
    extras_require=extras,
    setup_requires=['pytest-runner'],
    entry_points={
        'console_scripts': [
            'nixtract-nifti=nixtract.cli.nifti:main',
            'nixtract-gifti=nixtract.cli.gifti:main',
            'nixtract-cifti=nixtract.cli.cifti:main'
            ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering'
    ]
)
