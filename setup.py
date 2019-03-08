from setuptools import setup, find_packages

setup(
    name='vidhop',
    version='0.2',
    author='Florian Mock',
    py_modules=['vidhop'],
    packages=find_packages(),
    package_data={'vidhop': ['weights/*.pkl']},
    # include_package_data=True,
    install_requires=[
        'Click',
        'numpy',
        'keras',
        'numpy',
        'sklearn'
    ],
    entry_points={
        'console_scripts': [
            'vidhop =vidhop:cli'
        ]}

)

