from setuptools import setup, find_packages

setup(
    name="x-forecast",
    version="0.1.0",
    description="AI-Powered Demand Forecasting Engine",
    author="X-FORECAST Team",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.3.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)