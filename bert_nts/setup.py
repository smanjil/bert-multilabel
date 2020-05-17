import os

from setuptools import find_packages, setup


NAMESPACE = ''

d = os.path.join('src', NAMESPACE.replace('.', os.path.sep))

if NAMESPACE:
    p = [f'{NAMESPACE}.{pkg}' for pkg in find_packages(d)]
else:
    p = [pkg for pkg in find_packages(d)]


package_reqs = [
    'sklearn',
    'pandas',
    'transformers',
    'nltk',
    'torch',
]


setup(
    use_scm_version=True,
    scripts=[],
    setup_requires=['setuptools_scm'],
    install_requires=package_reqs,
    packages=p,
    include_package_data=True,
    package_dir={'': 'src'},
)

