from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    VERSION = _pkg_version("foambo")
except PackageNotFoundError:
    # Editable install before metadata is generated, or running from source
    # without install. Fall back to a sentinel so callers don't crash.
    VERSION = "0.0.0+unknown"

DEFAULT_CONFIG = "foamBO.yaml"
