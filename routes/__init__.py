"""Routes package for the application.

This module intentionally keeps the package import surface small. It
exposes the items blueprint so callers can import it as
`from routes.items import items_bp` or `from routes import items_bp`.
"""

from .items import items_bp  # re-export for convenience

__all__ = ["items_bp"]
