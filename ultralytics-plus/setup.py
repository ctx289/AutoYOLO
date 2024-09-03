# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
import pkg_resources as pkg

from setuptools import find_packages, setup

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / 'README.md').read_text(encoding='utf-8')
REQUIREMENTS = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements((PARENT / 'requirements.txt').read_text())]


# def get_version():
#     file = PARENT / 'ultralyticsp/__init__.py'
#     return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding='utf-8'), re.M)[1]


setup(
    name='ultralytics-plus',  # name of pypi package
    version='2.0.0',  # version of pypi package
    python_requires='>=3.9',
    license='AGPL-3.0',
    description=('Ultralytics-plus based on Ultralytics.'),
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/ultralytics/ultralytics',
    project_urls={
        'Bug Reports': 'https://github.com/ultralytics/ultralytics/issues',
        'Funding': 'https://ultralytics.com',
        'Source': 'https://github.com/ultralytics/ultralytics'},
    author='Ultralytics-plus',
    author_email='hello@ultralytics-plus.com',
    packages=find_packages(),  # required
    include_package_data=True,
    install_requires=REQUIREMENTS,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows', ],
    keywords='machine-learning, deep-learning, vision, ML, DL, AI, YOLO, YOLOv3, YOLOv5, YOLOv8, HUB, Ultralytics',)
