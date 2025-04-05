from setuptools import setup, find_packages

setup(
    name="fishing_sim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-openai>=0.0.2",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "faiss-cpu>=1.7.4",
    ],
    python_requires=">=3.8",
) 