import sys
import argparse
import os

# Ensure the project root is on the path so that `config`, `simulation`, etc.
# can be imported as top-level packages.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from config import SimConfig
from gui.main_window import MainWindow


def main():


    config = SimConfig()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow(config)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()