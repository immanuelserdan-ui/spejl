"""Spejl — mirror floor plans without mirroring the text.

``spejl`` is Danish for "mirror". The package name is also the pun: it is
the one word in the whole pipeline that must never be run through itself.
"""

try:
    # CI writes this module immediately before freezing a release. It is
    # intentionally absent from source checkouts, which use the fallback.
    from spejl._build_version import __version__
except ModuleNotFoundError:
    __version__ = "0.1.0"
