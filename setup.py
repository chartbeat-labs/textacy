import io
import os

from setuptools import setup, find_packages


about = {}
root_path = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(root_path, "textacy", "about.py")) as f:
    exec(f.read(), about)

# NOTE: Package configuration, including the name, metadata, and other options,
# are set in the setup.cfg file.
setup(
    version=about["__version__"],
    packages=find_packages(),
)
