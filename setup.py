import setuptools

setuptools.setup(
    name='error_metrics',
    version='0.0.1', 
    description='For time series forecasting error metrics',
    url='https://github.com/jaciz/error_metrics.git',
    author='Jaci',
    install_requires=['numpy'],
    author_email='jacquelinezhangg@gmail.com',
    packages=setuptools.find_packages(),
    zip_safe=False
)