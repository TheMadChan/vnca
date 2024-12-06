try:
    from setuptools._distutils.version import LooseVersion
except ImportError:
    from distutils.version import LooseVersion