"""PyInstaller entry point — imports the package properly so app.py's
relative imports resolve, instead of analyzing app.py as a top-level script."""

from squeak_peek.gui.app import main

if __name__ == "__main__":
    main()
