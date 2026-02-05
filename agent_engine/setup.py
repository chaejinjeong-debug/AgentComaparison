from setuptools import find_packages, setup

setup(
    name="agent_engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic-ai-slim[google]>=1.51.0",
        "google-cloud-aiplatform[agent_engines]>=1.78.0",
        "structlog>=24.0.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "opentelemetry-api>=1.20.0",
        "opentelemetry-sdk>=1.20.0",
        "opentelemetry-exporter-gcp-trace>=1.6.0",
        "opentelemetry-exporter-gcp-monitoring>=1.6.0",
        "google-cloud-trace>=1.11.0",
        "google-cloud-logging>=3.8.0",
        "google-cloud-monitoring>=2.18.0",
        "pyyaml>=6.0.3",
    ],
)
