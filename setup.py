from setuptools import setup, find_packages

setup(
    name='VIrus Deep learning HOst Prediction',
    version='0.2',
    author='Florian Mock',
    py_modules=['vidhop'],
    # py_modules=['cli','vidhop','DataParsing'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'numpy',
        'keras',
        'numpy',
        'sklearn'
    ],
    entry_points={
        'console_scripts': [
            'vidhop =cli:cli'
        ]}

)


'''
> MANIFEST.in tells Distutils what files to include in the source distribution
but it does not directly affect what files are installed. For that you need to
include the appropriate files in the setup.py file, generally either as package
data or as additional files. -- stackoverflow, 3596979

https://docs.python.org/3/distutils/setupscript.html#installing-package-data
https://docs.python.org/3/distutils/sourcedist.html#manifest
'''
