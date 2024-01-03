import sys
import importlib
import Georef3D_lib as lib
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

def create_viewer():
    global viewer
    viewer = lib.GeoReferencer()
    viewer.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Create a button to reload the module
   
    layout = QVBoxLayout()

    window = QWidget()
    window.setLayout(layout)
    window.show()

    create_viewer()  # Create the initial viewer

    sys.exit(app.exec_())