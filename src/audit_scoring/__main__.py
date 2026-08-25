"""Python module entrypoint: python -m audit_scoring"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
