import json
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import pandas as pd
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import (QMainWindow, QGraphicsView, QGraphicsScene, QVBoxLayout, QPushButton,
                             QWidget, QFileDialog, QLabel, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QInputDialog, QMenuBar, QMenu, QAction, QGraphicsEllipseItem, QGraphicsLineItem, QDialog, QComboBox,  QHBoxLayout, QCheckBox, QLineEdit, QSizePolicy, QSpacerItem, QSplitter)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush
from PyQt5.QtCore import Qt, pyqtSlot, QTimer, pyqtSignal
import copy


def calculateRMSE(valid_row, transformation_matrix):
    if transformation_matrix.shape == (4, 4):
        valid_row['rmse'] =  np.sqrt(
                np.sum(
                    (transform_uv_to_real(np.array([[valid_row['u'], valid_row['v']]]), transformation_matrix)[0] - 
                     np.array([valid_row['x'], valid_row['y'], valid_row['z']]))**2
                )
            )
    elif transformation_matrix.shape == (3, 3):
        valid_row['rmse'] =  np.sqrt(
                np.sum(
                    (transform_uv_to_real(np.array([[valid_row['u'], valid_row['v']]]), transformation_matrix)[0] - 
                     np.array([valid_row['x'], valid_row['y']]))**2
                )
            )
    return valid_row


def calculate_individual_rmse(row, transformation_matrix, transform_uv_to_real):
    """
    Calculate the Root Mean Square Error (RMSE) for an individual point.

    :param row: A dictionary representing a single row of data with 'u', 'v', 'x', 'y' (and 'z' if applicable).
    :param transformation_matrix: The transformation matrix.
    :param transform_uv_to_real: Function to transform image coordinates to real-world coordinates.
    :return: RMSE for the individual point.
    """

    # Extract image coordinates and world coordinates
    uv_coords = np.array([[row['u'], row['v']]])
    world_coords = np.array([row['x'], row['y']])

    # Transform image coordinates to world coordinates
    transformed_coords = transform_uv_to_real(uv_coords, transformation_matrix)

    # Calculate RMSE
    rmse = np.sqrt(np.sum((transformed_coords[0] - world_coords) ** 2))

    return rmse




def transform_uv_to_real(uv_coords, transformmatrix):
    if transformmatrix.shape == (4, 4):
        # Add a third and fourth dimension (W = 1 and homogeneous 1)
        uv_coords = np.hstack((uv_coords, np.ones((uv_coords.shape[0], 1)), np.ones((uv_coords.shape[0], 1))))
    elif transformmatrix.shape == (3, 3):
        # Add a third dimension (homogeneous 1)
        uv_coords = np.hstack((uv_coords, np.ones((uv_coords.shape[0], 1))))
    
    # Multiply the UV coordinates by the transformation matrix
    transformed_coords = np.dot(uv_coords, transformmatrix.T)
    #print('homogeneous coordinates: ', transformed_coords)    

    # Divide by the last column to get back to 2D or 3D coordinates
    if transformed_coords.shape[1] == 4:
        # Ensure the homogeneous coordinate is not zero
        if np.any(transformed_coords[:, 3] == 0):
            raise ValueError("Homogeneous coordinate is zero, cannot divide")
        #print('transformed_coords: ', transformed_coords[:, :3] / transformed_coords[:, 3, np.newaxis])
        return transformed_coords[:, :3] / transformed_coords[:, 3, np.newaxis]
    elif transformed_coords.shape[1] == 3:
        # Ensure the homogeneous coordinate is not zero
        if np.any(transformed_coords[:, 2] == 0):
            raise ValueError("Homogeneous coordinate is zero, cannot divide")
        #print('transformed_coords: ', transformed_coords[:, :2] / transformed_coords[:, 2, np.newaxis])
        return transformed_coords[:, :2] / transformed_coords[:, 2, np.newaxis]



def transform_real_to_uv(real_coords, transformmatrix):
    inv_matrix = np.linalg.inv(transformmatrix)
    # Depending weather the transformation matrix is 3x3 or 4x4, we need to use xy or xyz from the real-world coordinates.
    if transformmatrix.shape == (4, 4):
        real_coords = real_coords[:, :3]
    elif transformmatrix.shape == (3, 3):
        real_coords = real_coords[:, :2]
    # Add homogeneous dimension 1
    real_coords = np.hstack((real_coords, np.ones((real_coords.shape[0], 1))))
    # Multiply the real-world coordinates by the inverse transformation matrix
    imagecoords = np.dot(real_coords, inv_matrix.T)
    # Convert back to 2D image coordinates
    return imagecoords[:, :2] / imagecoords[:, 2:3]





def create_mesh_from_obb(obb):
    # Get the 8 corner points of the box
    corners = np.asarray(obb.get_box_points())
    
    # Define the 12 triangles using the indices of the corners
    triangles = [
        [0, 1, 2], [2, 3, 0],  # front
        [4, 5, 6], [6, 7, 4],  # back
        [0, 1, 5], [5, 4, 0],  # bottom
        [2, 3, 7], [7, 6, 2],  # top
        [0, 4, 7], [7, 3, 0],  # left
        [1, 5, 6], [6, 2, 1]   # right
    ]
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    return mesh

class CustomGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.geo_ref = parent  # Reference to the GeoReferencer
        self._pan = False
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        
        # Timer setup for distinguishing between click and pan
        self.click_timer = QTimer(self)
        self.click_timer.setSingleShot(True)
        self.click_timer.timeout.connect(self.start_panning)

    def start_panning(self):
        self._pan = True
        self.setCursor(Qt.ClosedHandCursor)

    def mousePressEvent(self, event):
        # For Panning
        if event.button() == Qt.LeftButton:
            self._pan_start_x = event.x()
            self._pan_start_y = event.y()
            # Start the timer when left button is pressed
            self.click_timer.start(200)  # 300 ms 

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # For Panning
        if self._pan:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - (event.x() - self._pan_start_x))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - (event.y() - self._pan_start_y))
            self._pan_start_x = event.x()
            self._pan_start_y = event.y()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.click_timer.stop()

        if not self._pan:
            scene_pos = self.mapToScene(event.pos())
            self.geo_ref.on_image_click(int(scene_pos.x()), int(scene_pos.y()))

        # Stop Panning
        if event.button() == Qt.LeftButton:
            self._pan = False
            self.setCursor(Qt.ArrowCursor)

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Zooming
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        # Save the scene pos
        old_pos = self.mapToScene(event.pos())

        # Zoom
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)

        # Get the new position
        new_pos = self.mapToScene(event.pos())

        # Move scene to old position
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def draw_reference_point(self, x, y, color=Qt.red):
        """Draw a reference point as a vector ellipse on the QGraphicsScene."""
        if x is not None and y is not None:
            if isinstance(x, (float)) and isinstance(y, (float)):
                ellipse_item = self.scene().addEllipse(x-3, y-3, 6, 6, QPen(color), QBrush(color))
                return ellipse_item
            else:
                print(f"Skipping drawing point at ({x}, {y}) due to invalid coordinates.")
        else:
            print(f"Skipping drawing point at ({x}, {y}) due to None value.")
        #def draw_error_line(self, start_point, end_point, color=Qt.red):
           #"""Draw an error line between two points on the QGraphicsScene."""
           #line_item = self.scene().addLine(start_point[0], start_point[1], end_point[0], end_point[1], QPen(color))
            #return line_item
    

class CSVStructureDialog(QDialog):
    # Define a signal to emit the selected data
    data_imported = pyqtSignal(list)
    def __init__(self, csv_data=None):
        super().__init__()

        self.csv_data = csv_data
        self.mapping = {}
        self.skip_rows = 0  # Default value for skipping rows

        # Set a reasonable default window size
        self.setGeometry(100, 100, 800, 400)

        layout = QVBoxLayout(self)

        # Skip Rows Input
        skip_rows_layout = QHBoxLayout()
        skip_rows_label = QLabel("Skip Rows:", self)
        self.skip_rows_input = QLineEdit(self)
        self.skip_rows_input.setText(str(self.skip_rows))
        self.skip_rows_input.setMaximumWidth(40)  # Set a maximum width of 40 pixels
        self.skip_rows_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Fix the size
        self.skip_rows_input.textChanged.connect(self.update_skip_rows)
        skip_rows_layout.addWidget(skip_rows_label)
        skip_rows_layout.addWidget(self.skip_rows_input)
        layout.addLayout(skip_rows_layout)

        # Create the table with an additional column for checkboxes and a row for mapping selection
        self.table = QTableWidget(self)
        self.table.setColumnCount(len(csv_data[0]) + 1)  # Additional column for checkboxes
        self.table.setRowCount(len(csv_data) - self.skip_rows + 1)  # Additional row for mapping selection

        self.row_checkboxes = []  # Store row selection checkboxes
        self.checkbox_states = []  # Store checkbox states

        # Add checkboxes in the additional column (column 0) and mapping selection in the additional row (row 0)
        for row, row_data in enumerate(csv_data[1 + self.skip_rows:], start=0):  # Skip the header and skipped rows
            # Add a checkbox for row selection
            checkbox = QCheckBox(self)
            self.row_checkboxes.append(checkbox)
            self.checkbox_states.append(False)  # Initialize all checkboxes as unchecked
            self.table.setCellWidget(row, 0, checkbox)  # In the extra column (column 0)
            checkbox.stateChanged.connect(lambda state, row=row: self.checkbox_state_changed(state, row))  # Connect the state change signal

            for col, item in enumerate(row_data):
                # Set table item
                table_item = QTableWidgetItem(item)
                self.table.setItem(row, col + 1, table_item)  # Shift by 1 to accommodate the extra column

        # Set headers (including the additional column)
        header_labels = [""] + csv_data[0]  # Empty label for the additional column
        self.table.setHorizontalHeaderLabels(header_labels)

        # Add mapping selection dropdowns in the first row (above the header, excluding the additional column)
        for col, header in enumerate(csv_data[0]):
            combo = QComboBox(self)
            combo.addItems(["-", "name", "u", "v", "x", "y", "z"]) 
            combo.currentIndexChanged.connect(lambda idx, col=col, combo=combo: self.update_mapping(col, combo))
            combo.setFixedWidth(self.table.columnWidth(col + 1))  # Set the width of the combo box to match the column width
            self.table.setCellWidget(0, col + 1, combo)  # Add to the first row (shifted by 1)

        layout.addWidget(self.table)

        # "Select All" checkbox (top-right corner with a margin)
        select_all_layout = QHBoxLayout()
        select_all_checkbox = QCheckBox("Select All", self)
        select_all_checkbox.setChecked(False)  # Uncheck by default
        select_all_checkbox.stateChanged.connect(self.select_all_rows)
        select_all_layout.addItem(QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Fixed))  # Add margin
        select_all_layout.addWidget(select_all_checkbox)
        layout.addLayout(select_all_layout)

        # Import button
        btn_import = QPushButton("Import Ref Points", self)
        btn_import.clicked.connect(self.on_import)
        layout.addWidget(btn_import)

        self.setLayout(layout)

    def update_mapping(self, col, combo):
        value = combo.currentText()
        if value != "-":
            self.mapping[value] = col
    


    def on_import(self):
        # Find selected rows based on checkbox_states
        selected_rows = [i for i, state in enumerate(self.checkbox_states) if state]
        print('selected_rows: ', selected_rows)

        # Initialize the imported_data list
        imported_data = []

        # Iterate through the selected rows and check the corresponding checkboxes
        for row in selected_rows:
            print(f"Row {row} is selected.")
            #if self.row_checkboxes[row].isChecked():
            print(f"Row {row} is checked.")
            row_data = []
            for col in range(1, self.table.columnCount()):  # Start from column 1 to exclude the checkbox column
                item = self.table.item(row, col)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append("")  # Handle empty cellss
            imported_data.append(row_data)
        print(imported_data)
        # Emit the signal with the imported data
        self.data_imported.emit(imported_data)
        self.accept()  # Use self.reject() if you want to cancel the changes

    def select_all_rows(self, state):
        # Set checkbox states for all rows based on the "Select All" checkbox state
        self.checkbox_states = [state == Qt.Checked] * len(self.row_checkboxes)
        # Update the checkboxes to reflect the state change
        for checkbox in self.row_checkboxes:
            checkbox.setChecked(state == Qt.Checked)

    def checkbox_state_changed(self, state, row):
        # Update the checkbox state in the list when a checkbox is clicked
        self.checkbox_states[row] = state == Qt.Checked

    def update_skip_rows(self):
        # Update the number of rows to skip based on user input
        try:
            self.skip_rows = int(self.skip_rows_input.text())
        except ValueError:
            self.skip_rows = 0  # Default to 0 if input is not a valid integer

        # Adjust the table row count based on the new skip_rows value
        self.table.setRowCount(len(self.csv_data) - self.skip_rows + 1)

        # Reset the checkbox states and mappings when the number of skipped rows changes
        self.checkbox_states = [False] * len(self.row_checkboxes)
        self.mapping = {}

class GeoReferencer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_data = {}
        self.image = None
        self.original_image = None
        self.image_path = ""
        self.reference_points = pd.DataFrame()
        #self.reference_points = self.reference_points.astype({'name': 'str'  ,'u': 'float', 'v': 'float', 'x': 'float', 'y': 'float', 'z': 'float', 'rmse': 'float'})
        self.mesh = None
        self.editing_uv_for_row = None


        self.config_file = "config.json" 


        self.initUI()

    def initUI(self):
        self.setWindowTitle("3D Georeferencer")

        # Menu Bar
        menu_bar = QMenuBar(self)
        file_menu = QMenu("File", self)
        options_menu = QMenu("Options", self)

        load_image_action = QAction('Load Image', self)
        load_image_action.triggered.connect(self.load_image)
        

        file_menu.addAction(load_image_action)

        load_ref_points_action = QAction('Load Reference Points', self)
        load_ref_points_action.triggered.connect(self.load_reference_points)
        file_menu.addAction(load_ref_points_action)
        


        save_ref_points_action = QAction('Save Reference Points', self)
        save_ref_points_action.triggered.connect(self.save_reference_points)
        file_menu.addAction(save_ref_points_action)

        export_mesh_action = QAction('Export Mesh', self)
        export_mesh_action.triggered.connect(self.export_mesh)
        file_menu.addAction(export_mesh_action)

        settings_action = QAction('Settings', self)
        settings_action.triggered.connect(self.create_or_modify_config)
        options_menu.addAction(settings_action)

        menu_bar.addMenu(file_menu)
        menu_bar.addMenu(options_menu)
        self.setMenuBar(menu_bar)

        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)

        # Image path label
        self.image_path_label = QLabel("No image loaded", self)
        layout.addWidget(self.image_path_label)



        # GraphicsView for Image
        self.view = CustomGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        layout.addWidget(self.view)

        # Reference table
        self.table = QTableWidget(self)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setColumnCount(7)  # Change to 7 columns
        self.table.setHorizontalHeaderLabels(["name", "u", "v", "x", "y", "z", "rmse"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        self.table.cellChanged.connect(self.on_table_cell_changed)
        self.table.cellDoubleClicked.connect(self.on_table_cell_double_clicked)


        # Create a splitter
        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Vertical)

        # Add GraphicsView and Table to Splitter
        splitter.addWidget(self.view)
        splitter.addWidget(self.table)

        # Add Splitter to Layout
        layout.addWidget(splitter)
        
        # Buttons

        self.calc_transform_buttonXY = QPushButton("Calculate Transformation XY", self)
        self.calc_transform_buttonXY.clicked.connect(self.compute_projective_transformation_xy)
        layout.addWidget(self.calc_transform_buttonXY)

        self.calc_transform_buttonXYZ = QPushButton("Calculate Transformation XYZ", self)
        self.calc_transform_buttonXYZ.clicked.connect(self.compute_projective_transformation_xyz)
        layout.addWidget(self.calc_transform_buttonXYZ)

        # Add a button for a function that conducts the transformation and fills in x,y,z from u,v. 
        self.fillEmptyXYDimensions_button = QPushButton("fillEmptyXYDimensions", self)
        self.fillEmptyXYDimensions_button.clicked.connect(self.fillEmptyXYDimensions)
        layout.addWidget(self.fillEmptyXYDimensions_button)

        self.view_3d_button = QPushButton("View in 3D", self)
        self.view_3d_button.clicked.connect(self.view_in_3d)
        layout.addWidget(self.view_3d_button)

        self.reset_button = QPushButton("Reset", self)
        self.reset_button.clicked.connect(self.reset_image)
        layout.addWidget(self.reset_button)

        # Add a button to create the plane and box from reference points
        self.create_plane_box_button = QPushButton("Create Plane and Box", self)
        self.create_plane_box_button.clicked.connect(self.create_plane_and_box_with_corners_action)
        layout.addWidget(self.create_plane_box_button)

        self.textured_mesh_button = QPushButton("Create Textured Mesh", self)
        self.textured_mesh_button.clicked.connect(self.create_textured_mesh)
        layout.addWidget(self.textured_mesh_button)

        self.setCentralWidget(central_widget)
        self.setGeometry(100, 100, 800, 600)
    
    def handle_imported_data(self, imported_data):
        templistofdicts = []

        for row_data in imported_data:

            refpoint = {}
            try:
                refpoint['name'] = row_data[self.dialog.mapping.get("name", -1)] if "name" in self.dialog.mapping else None
                refpoint['u'] = float(row_data[self.dialog.mapping.get("u", -1)]) if "u" in self.dialog.mapping else None
                refpoint['v'] = float(row_data[self.dialog.mapping.get("v", -1)]) if "v" in self.dialog.mapping else None
                refpoint['x'] = float(row_data[self.dialog.mapping.get("x", -1)]) if "x" in self.dialog.mapping else None
                refpoint['y'] = float(row_data[self.dialog.mapping.get("y", -1)]) if "y" in self.dialog.mapping else None
                refpoint['z'] = float(row_data[self.dialog.mapping.get("z", -1)]) if "z" in self.dialog.mapping else None
                refpoint['rmse'] = None
                templistofdicts.append(refpoint)

            except ValueError as e:
                print(f"Invalid data: {e}")
                continue
            #print(pd.DataFrame(templistofdicts))
        self.reference_points = pd.concat([self.reference_points, pd.DataFrame(templistofdicts) ], ignore_index=True)


    @pyqtSlot()

    def update_table_with_dataframe(self):
        self.table.blockSignals(True)
        self.table.setRowCount(self.reference_points.shape[0])
        self.table.setColumnCount(self.reference_points.shape[1])
        self.table.setHorizontalHeaderLabels(self.reference_points.columns)

        for i, row in self.reference_points.iterrows():
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value) if value else "")
                self.table.setItem(i, j, item)
        self.table.blockSignals(False)
    @pyqtSlot()
    def load_reference_points(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select a JSON or CSV file", "", "JSON files (*.json);;CSV files (*.csv);;All Files (*)")

        if filepath.endswith(".json"):
            with open(filepath, 'r') as file:
                data = json.load(file)
                #uprint(data)

            self.reference_points = pd.concat([self.reference_points, pd.DataFrame(data) ], ignore_index=True)


        elif filepath.endswith(".csv"):
            import csv
            with open(filepath, 'r') as file:
                reader = csv.reader(file)
                csv_data = list(reader) 

            self.dialog = CSVStructureDialog(csv_data)

            # Connect the signal to the slot here
            self.dialog.data_imported.connect(self.handle_imported_data)
            result = self.dialog.exec_()


        self.update_reference_and_corner_points_on_canvas()
        self.update_table_with_dataframe()
    

    @pyqtSlot()
    def save_reference_points(self):
        if self.reference_points.empty:
            print("No reference points to save.")
            return

        default_filename = self.image_path.rsplit('.', 1)[0] + '.json'
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Reference Points", default_filename, "JSON files (*.json);;All Files (*)")
        #referencepoints = {'reference_points': self.reference_points.to_dict(orient='records')}
        records = self.reference_points.to_dict(orient='records')
        print(records)

        if filepath:
            with open(filepath, 'w') as file:
                json.dump(records, file)
            print(f"Reference points saved to {filepath}")




    def compute_projective_transformation_xyz(self):
        """
        Compute the 3D-to-3D projective transformation matrix using the DLT method.
        Parameters:
        -self.reference_points: Nx7 pandas DataFrame of reference points.
        - 4x4 projective transformation matrix.
        """
        # select from reference points only the rows that have valid float values not None nor Nan in the columns u,v,x,y,z.
        valid_rows_3D = self.reference_points[(self.reference_points['u'].notnull()) &
                                           (self.reference_points['v'].notnull()) &
                                           (self.reference_points['x'].notnull()) &
                                           (self.reference_points['y'].notnull()) &
                                           (self.reference_points['z'].notnull())].copy()
        print('so many valid rows: ', len(valid_rows_3D))
        
        # Check if there are at least 4 valid reference points
        if len(valid_rows_3D) < 4:
            raise ValueError("At least 4 reference points are required.")

        
        # Construct the matrix A.
        A = []
        for i, row in valid_rows_3D.iterrows(): 
            u = row['u']
            v = row['v']
            w = 1.0
            x = row['x']
            y = row['y']
            z = 10.0
            A.append([u, v, w, 1, 0, 0, 0, 0, 0, 0, 0, 0, -u*x, -v*x, -w*x, -x])
            A.append([0, 0, 0, 0, u, v, w, 1, 0, 0, 0, 0, -u*y, -v*y, -w*y, -y])
            A.append([0, 0, 0, 0, 0, 0, 0, 0, u, v, w, 1, -u*z, -v*z, -w*z, -z])            
        A = np.array(A)
        #print('A: ', A)
        # Singular Value Decomposition to solve for the transformation matrix.
        U, S, Vt = np.linalg.svd(A)
        self.transformation_matrix_xyz = Vt[-1].reshape(4, 4)
        valid_rows_3D = valid_rows_3D.apply(calculateRMSE, transformation_matrix = self.transformation_matrix_xyz, axis=1)
        print(self.transformation_matrix_xyz)
        #Update the original self.reference_points with the calculated RMSE
        self.reference_points.update(valid_rows_3D)

        #update the table widget
        self.update_table_with_dataframe()
    
    def compute_projective_transformation_xy(self):
        """
        Compute the 2D-to-2D projective transformation matrix using the DLT method.
        Parameters:
        -self.reference_points: Nx7 pandas DataFrame of reference points.
        """
        valid_rows_2D = self.reference_points[(self.reference_points['u'].notnull()) &
                                            (self.reference_points['v'].notnull()) &
                                            (self.reference_points['x'].notnull()) &
                                            (self.reference_points['z'].notnull()) &
                                            (self.reference_points['y'].notnull())].copy()
        print('so many valid rows: ', len(valid_rows_2D))
        if len(valid_rows_2D) < 4:
            raise ValueError("At least 4 reference points are required.")
        
        # Construct the matrix A.
        A = []
        for i, row in valid_rows_2D.iterrows():
            u = row['u']
            v = row['v']
            x = row['x']
            y = row['y']
            A.append([u, v, 1, 0, 0, 0, -u*x, -v*x, -x])
            A.append([0, 0, 0, u, v, 1, -u*y, -v*y, -y])
        A = np.array(A)
        print(A)
        # Singular Value Decomposition to solve for the transformation matrix.
        U, S, Vt = np.linalg.svd(A)
        self.transformation_matrix_xy = Vt[-1].reshape(3, 3)
        valid_rows_2D = valid_rows_2D.apply(calculateRMSE, transformation_matrix = self.transformation_matrix_xy, axis = 1)
        print(self.transformation_matrix_xy)
        #Update the original self.reference_points with the calculated RMSE
        self.reference_points.update(valid_rows_2D)

        #update the table widget
        self.update_table_with_dataframe()


    def fillEmptyXYDimensions(self):
        transformation_matrix=self.transformation_matrix_xy
        # select from all reference points only those which have valid float values in u and v columns and empty values in x and y columns.
        valid_rows = self.reference_points[(self.reference_points['u'].notnull()) &
                                             (self.reference_points['v'].notnull()) &
                                                (self.reference_points['x'].isnull()) &
                                                (self.reference_points['y'].isnull())].copy()
        print(valid_rows)

        # Check if there is a transformation matrix with 3x3 dimensions for XY-transformation.
        if transformation_matrix.shape != (3, 3):
            raise ValueError("Transformation matrix must be 3x3.")
        # sel.reference_points is a pandas dataframe with columns u,v,x,y,z. Use the function transform_uv_to_real to fill in the empty x,y columns.

        for i, row in valid_rows.iterrows():
            u = row['u']
            v = row['v']
            uv_coords = np.array([[u, v]])
            real_coords = transform_uv_to_real(uv_coords, transformation_matrix)
            self.reference_points.loc[i, ['x', 'y']] = real_coords

        # Update the table widget
        self.update_table_with_dataframe()

    @pyqtSlot()
    def export_mesh(self):
        if self.config_data:
            if self.config_data['translation_on_export']:
                export_mesh = copy.deepcopy(self.mesh)
                export_mesh.translate(self.config_data['translation_on_export'])
                print(f"Mesh translated to {export_mesh.get_center()}.")
        # Check if mesh exists
        if not hasattr(self, 'mesh') or not self.mesh.vertices:
            print("No mesh available for export.")
            return

        # Open a file dialog for the user to select a save location
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Mesh", "", "OBJ files (*.obj);;All Files (*)")
        if not filepath:
            return

        # If the selected filepath doesn't have the .obj extension, add it
        if not filepath.lower().endswith(".obj"):
            filepath += ".obj"

        # Export the mesh
        #texture = self.original_image if self.original_image else None  # Use the original image as texture
        o3d.io.write_triangle_mesh(filepath, export_mesh)
        #export_textured_mesh(self.mesh, texture, filepath)
        print(f"Mesh exported to {filepath}")

    @pyqtSlot()
    def create_plane_and_box_with_corners_action(self):

        self.create_plane_and_box_with_corners()
        return self



    def create_or_modify_config(self):
        # Open a file dialog to ask the user to select a config.json file
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        config_filepath, _ = QFileDialog.getOpenFileName(self, "Select config.json file", "", "JSON files (*.json);;All Files (*)", options=options)

        # If the user selected a file, load it
        if config_filepath:
            with open(config_filepath, "r") as file:
                self.config_data = json.load(file)
            print(f"Loaded config from {config_filepath}")


    def config_file_exists(self):
        try:
            with open(self.config_file, "r") as file:
                return True
        except FileNotFoundError:
            return False



    def on_image_click(self, u, v):
        u = float(u)
        v = float(v)
        if hasattr(self, 'editing_uv_for_row') and self.editing_uv_for_row is not None:
            self.reference_points.loc[self.editing_uv_for_row, 'u'] = u
            self.reference_points.loc[self.editing_uv_for_row, 'v'] = v
            delattr(self, 'editing_uv_for_row')

        else:
            # Store clicked image coordinates
            new_point = {'u': u, 'v': v, 'x': None, 'y': None, 'z': None, 'rmse': None}

            # Prompt user for real-world coordinates
            coords, ok = QInputDialog.getText(self, "Input", "Enter real-world coordinates x, y, z with comma-seperator:")

            if ok and coords:
                coords_list = coords.split(',')
                if len(coords_list) != 3:
                    print("Invalid coordinates.")
                    return
                # replace empty strings in coords_list with None.
                coords_list = [float(coord.strip()) if coord.strip() else None for coord in coords_list]
                new_point['x'] = coords_list[0]
                new_point['y'] = coords_list[1]
                new_point['z'] = coords_list[2]

                self.reference_points = pd.concat([self.reference_points, pd.DataFrame([new_point])], ignore_index=True)
        self.update_table_with_dataframe()
        self.update_reference_and_corner_points_on_canvas()
        

    def ask_for_image_file(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "JPEG files (*.jpg;*.jpeg);;PNG files (*.png);;All Files (*)", options=options)
        return filepath

    @pyqtSlot()
    def load_image(self):
        ## developer setup
        self.image_path = self.ask_for_image_file()
        #self.image_path = "C:/Users/tronc/Nextcloud/Uruk/WES_paleoenvi/URUK_ERT/Uruk_2023_03_18_Profile_11_SN_Schlumberger_cut.jpeg"
        if not self.image_path:
            return

        self.image = Image.open(self.image_path).convert('RGBA')
        self.original_image = self.image.copy()

        # Update label
        self.image_path_label.setText(f"Loaded Image: {self.image_path}")

        # Display image
        qim = QImage(self.image.tobytes("raw", "RGBA"), self.image.width, self.image.height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qim)
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.view.setScene(self.scene)
    
    def on_table_cell_double_clicked(self, row, column):
        if column in [1,2]:  # Assuming u,v is column 1
            self.editing_uv_for_row = row  # Create this attribute in __init__
            #self.view.scene().removeItem(self.reference_points[]  # Assuming you store QGraphicsItem for each point

    def on_table_cell_changed(self, row, column):
        value = self.table.item(row, column).text()
        print(f"Cell ({row}, {column}) changed to {value}")
        if value == "":
            self.reference_points.iloc[row, column] = None
        if column == 0:
            self.reference_points.iloc[row, column] = str(value)
        if column > 0 and value != "":
            self.reference_points.iloc[row, column] = float(value)
        #except ValueError:
            #self.table.item(row, column).setText("")  # Reset the cell to empty if the value is invalid

        self.update_reference_and_corner_points_on_canvas()






    @pyqtSlot()
    def reset_image(self):
        self.image = self.original_image.copy()
        self.image_coordinates = []
        self.real_world_coordinates = []

        # Display the original image
        qim = QImage(self.image.tobytes("raw", "RGBA"), self.image.width, self.image.height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qim)
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.view.setScene(self.scene)

    @pyqtSlot()
    def create_textured_mesh(self):
        if not self.real_world_coordinates:
            print("No reference points available.")
            return

        # 1. Call create_plane_and_box_with_corners to get the point cloud
        box_pcd_utm, box = create_plane_and_box_with_corners(self.image_coordinates, self.real_world_coordinates, self.image_path, self.original_image)
        #box_pcd_utm.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=10))
        hull, _ =box_pcd_utm.compute_convex_hull(joggle_inputs=True)
        #o3d.visualization.draw_geometries([box_pcd_utm, hull ], mesh_show_wireframe=True, point_show_normal=True)
        print("Number of points:", len(hull.vertices))
        print("Number of triangles:", len(hull.triangles))

        # 2. Create a triangle mesh using the Poisson reconstruction method
        depth = 8  # You can adjust these parameters based on your requirements
        scale = 4
        mesh = hull


        # Check if the mesh is valid
        if not mesh.vertices:
            print("Mesh generation failed. Please ensure there are enough reference points and they are well-distributed.")
            return

        print("Number of vertices:", len(mesh.vertices))
        

        #self.mesh = mesh
        # Visualize the mesh
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh.compute_uvatlas()
        #size = 512, gutter = 1.0, max_stretch = 0.1666666716337204, parallel_partitions  = 1, nthreads  = 0
        # Generate UV coordinates from image_coordinates
        uv_coordinates = np.array(self.image_coordinates, dtype=np.float64)
        uv_coordinates[:, 0] /= float(self.image.width)   # Normalize U coordinates
        uv_coordinates[:, 1] = 1.0 - uv_coordinates[:, 1] / float(self.image.height)  # Normalize V coordinates and flip vertically
        
        # Assign UV coordinates to the mesh
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_coordinates)
        
        # Read the image as a texture
        texture = o3d.io.read_image(self.image_path)
        
        # Assign texture to the mesh
        mesh.textures = [texture]
        o3d.visualization.draw_geometries([mesh, box_pcd_utm], mesh_show_wireframe=True, point_show_normal=True)
 
    
    def view_in_3d(self):
        # Create a PointCloud object from the reference points
        pcd = o3d.geometry.PointCloud()
        valid_rows = self.reference_points[(self.reference_points['z'].notnull()) &
                                        (self.reference_points['x'].notnull()) &
                                        (self.reference_points['y'].notnull())].copy()
        

        pcd.points = o3d.utility.Vector3dVector(np.array(valid_rows[['x', 'y', 'z']]))
        print(np.asarray(pcd.points))

        # Check if the PointCloud has normals, if not compute them
        #if not pcd.has_normals():
            #pcd.estimate_normals()

        # Visualization objects
        vis_objects = [pcd]

        # If a mesh exists, add it to the visualization
        #if hasattr(self, 'mesh') :
            #vis_objects.append(self.mesh)

        o3d.visualization.draw_geometries(vis_objects)
    


    def create_plane_and_box_with_corners(self):

        img = o3d.io.read_image(self.image_path)
        img_width, img_height = np.asarray(img).shape[1], np.asarray(img).shape[0]
        #select from reference points only the rows that have valid float values not None nor Nan in the columns x,y,z.
        valid_rows = self.reference_points[(self.reference_points['x'].notnull()) &
                                             (self.reference_points['y'].notnull()) &
                                                (self.reference_points['z'].notnull())].copy()
        

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(valid_rows[['x', 'y', 'z']]))

        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=100)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal) 
        print(f"Plane normal: {normal}")
        corner_uv_coords = np.array([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
        corner_xyz_coords = transform_uv_to_real(corner_uv_coords, self.transformation_matrix_xyz)

        image_corners_xyz = corner_xyz_coords
        corner_pcd = o3d.geometry.PointCloud()
        corner_pcd.points = o3d.utility.Vector3dVector(image_corners_xyz)
        extruded_image_corners = copy.deepcopy(corner_pcd).translate((-self.config_data['frame_thickness']* normal[0], -self.config_data['frame_thickness'] * normal[1], -self.config_data['frame_thickness'] * normal[2]), relative=True)

        normals_for_ref = np.tile(normal, (len(valid_rows), 1))
        pcd.normals = o3d.utility.Vector3dVector(normals_for_ref)

        normals_for_corner = np.tile(normal, (len(image_corners_xyz), 1))
        corner_pcd.normals = o3d.utility.Vector3dVector(normals_for_corner)

        inverted_normal = -normal
        normals_for_extruded_corners = np.tile(inverted_normal, (len(image_corners_xyz), 1))
        extruded_image_corners.normals = o3d.utility.Vector3dVector(normals_for_extruded_corners)

        box_pcd_utm = pcd + corner_pcd + extruded_image_corners

        # Find orthogonal vectors to the normal
        if abs(normal[0]) > abs(normal[1]):
            u = np.cross(normal, [0, 1, 0])
        else:
            u = np.cross(normal, [1, 0, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

        # Form the rotation matrix and rotate the points
        R = np.vstack([u, v, normal])
        rotated_points = np.dot(np.asarray(box_pcd_utm.points), R.T)

        # 3. Compute AABB in the rotated space
        min_pt = np.min(rotated_points, axis=0)
        max_pt = np.max(rotated_points, axis=0)

        # 4. Rotate the AABB back to the original space to get the OBB
        obb_corners = np.array([
            [min_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], max_pt[1], max_pt[2]],
            [min_pt[0], max_pt[1], max_pt[2]]
        ])
        obb_corners = np.dot(obb_corners, np.linalg.inv(R.T))
        obb_corners_vector = o3d.utility.Vector3dVector(obb_corners)
        box = o3d.geometry.OrientedBoundingBox.create_from_points(obb_corners_vector)
    

        
        # Convert OrientedBoundingBox to TriangleMesh with UV mapping
        box_mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(box, create_uv_map=True)
        num_of_vertices = len(box_mesh.vertices)
        red_color = [1, 0, 0]  # RGB for red
        #box_mesh.vertex_colors = o3d.utility.Vector3dVector([red_color] * num_of_vertices)
        # 1. Identify the face to be textured

        # Compute face normals of the mesh
        box_mesh.compute_triangle_normals(normalized=True)
        face_normals = np.asarray(box_mesh.triangle_normals)

        # Identify the two faces that are aligned with the normal
        dot_products = np.dot(face_normals, normal)
        sorted_indices = np.argsort(dot_products)
        target_face_indices = sorted_indices[-2:]  # Get the last two indices (highest dot products)

        # Identify the two faces that are opposite with the normal
        dot_products = np.dot(face_normals, -1 * normal)
        sorted_indices = np.argsort(dot_products)
        opposite_face_indices = sorted_indices[-2:]  # Get the last two indices (highest dot products)

        print(f"Target face indices: {target_face_indices}")

        # UV coordinates for the entire texture
        full_texture_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        # For simplicity, other vertices will map to a tiny portion of the texture (e.g., top-left corner)
        tiny_texture_uv = np.array([[0, 0]])



        print(f"Target face indices: {np.append(target_face_indices,opposite_face_indices)}")
        # Gather all unique vertices from the target faces
        unique_vertices = set()
        for target_face_idx in np.append(target_face_indices,opposite_face_indices):
            face_vertices = np.asarray(box_mesh.triangles)[target_face_idx]
            unique_vertices.update(face_vertices)


        # Create a mapping of vertex indices to their UV coordinates
        vertex_to_uv = {}

        for vert_idx, vert in enumerate(box_mesh.vertices):
            if vert_idx in unique_vertices:
                xyz = np.asarray(vert)
                uv = transform_real_to_uv(np.array([xyz]), self.transformation_matrix_xyz)[0]
                u_normalized = uv[0] / img_width
                v_normalized =  (uv[1] / img_height)
                vertex_to_uv[vert_idx] = [u_normalized, v_normalized]
            else:
                vertex_to_uv[vert_idx] = [0, 0]  # default value

        # Construct UV coordinates for every vertex of every triangle
        triangle_uv_coords = []
        for triangle in box_mesh.triangles:
            for vert_idx in triangle:
                triangle_uv_coords.append(vertex_to_uv[vert_idx])

        print('Triangle UV coords', triangle_uv_coords)

        # Assign the UV coordinates to the mesh
        box_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uv_coords)

        # Apply the texture

        box_mesh.textures = [img]
        num_points = len(pcd.points)
        black_color = [0, 0, 0]  # RGB for black
        pcd.colors = o3d.utility.Vector3dVector([black_color] * num_points)

        # For visualization using the material
        material = rendering.MaterialRecord()
        material.shader = 'defaultUnlit'
        material.albedo_img = img
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=box_mesh.get_center())
        frame.translate((10, 10, 20))
        
        # Create a visualization window
        #vis = o3d.visualization.Visualizer()
        #vis.create_window()
        # Store the mesh in the instance variable
        self.mesh = box_mesh
        o3d.visualization.draw([{'name': 'box', 'geometry': box_mesh, 'material': material}, frame, {'name': 'refpoints', 'geometry': pcd, 'material': material, 'point_show_normal': True}]) 

        # Add the point cloud
        #vis.add_geometry(pcd)

        return box_pcd_utm, box_mesh


    def update_reference_and_corner_points_on_canvas(self):
        """
        Update the reference and corner points on the canvas.
        """
        # Clear the previous vector items from the scene
        for item in self.view.scene().items():
            if isinstance(item, (QGraphicsEllipseItem, QGraphicsLineItem)):
                self.view.scene().removeItem(item)

        # Get transformed UV coordinates using the transformation matrix
        #if hasattr(self, 'transformation_matrix'):  # Ensure transformation matrix is available
            #transformed_uv = transform_real_to_uv(np.array(self.reference_points[['x']]), self.transformation_matrix)
            #corners_uv = transform_real_to_uv(transform_image_corners(self.original_image, self.transformation_matrix), self.transformation_matrix)
        #else:
            #transformed_uv = self.image_coordinates  # Default to original if no transformation matrix available
            #corners_uv = [(0, 0), (self.original_image.width, 0), (0, self.original_image.height), (self.original_image.width, self.original_image.height)]

        # Draw points
        for i, row in self.reference_points.iterrows():
            if row['u'] is None or row['v'] is None:
                continue

            # Original user-input points
            self.view.draw_reference_point(float(row['u']), float(row['v']), color=Qt.red)
            # Transformed points
            #self.view.draw_reference_point(trans[0], trans[1], color=Qt.blue)
            # Error line
            #self.view.draw_error_line(orig, trans, color=Qt.red)

        # Draw corner points in green
        #for corner in corners_uv:
            #self.view.draw_reference_point(corner[0], corner[1], color=Qt.green)
