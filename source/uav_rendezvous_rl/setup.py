"""Installation script for the uav_rendezvous_rl package."""

from setuptools import find_packages, setup


setup(
    name="uav_rendezvous_rl",
    version="0.1.0",
    description="Minimal Isaac Lab external project for UAV rendezvous RL milestones.",
    author="lab_726",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.11",
    zip_safe=False,
)
