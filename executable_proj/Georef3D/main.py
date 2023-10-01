import sys
import importlib
import Georef3D_lib as lib
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

def reload_module():
    global lib
    importlib.reload(lib)
    # You may also want to recreate the viewer after reloading the module
    viewer.close()  # Close the old viewer
    create_viewer()  # Create a new viewer

def create_viewer():
    global viewer
    viewer = lib.GeoReferencer()
    viewer.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Create a button to reload the module
    reload_button = QPushButton("Reload Module")
    reload_button.clicked.connect(reload_module)

    layout = QVBoxLayout()
    layout.addWidget(reload_button)

    window = QWidget()
    window.setLayout(layout)
    window.show()

    create_viewer()  # Create the initial viewer

    sys.exit(app.exec_())