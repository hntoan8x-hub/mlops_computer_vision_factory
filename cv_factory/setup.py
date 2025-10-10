from setuptools import setup, find_packages

setup(
    name='cv-factory',
    version='0.1.0',
    packages=find_packages(where='cv_factory'),
    package_dir={'': 'cv_factory'},
    install_requires=[
        # Core libraries
        'numpy>=1.24.0',
        'PyYAML>=6.0',
        'pydantic>=1.10.0',
        # ... (thêm các thư viện khác từ requirements.txt vào đây)
    ],
    author='Your Name',
    description='An MLOps framework for Computer Vision applications.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-username/cv-factory',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha'
    ],
    python_requires='>=3.10',
)