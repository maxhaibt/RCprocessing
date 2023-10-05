import sys
import json
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import QApplication

from PyQt5.QtWidgets import (QMainWindow, QGraphicsView, QGraphicsScene, QVBoxLayout, QPushButton,
                             QWidget, QFileDialog, QLabel, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QInputDialog, QMenuBar, QMenu, QAction, QGraphicsEllipseItem, QGraphicsLineItem, QDialog, QComboBox,  QHBoxLayout)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
import copy



def compute_projective_transformation(uv_coords, xyz_coords):
    """
    Compute the 3D-to-3D projective transformation matrix using the DLT method.
    
    Parameters:
    - uv_coords: List of 2D points. Each point is a tuple (u, v).
    - xyz_coords: List of corresponding 3D points. Each point is a tuple (x, y, z).
    
    Returns:
    - 4x4 projective transformation matrix.
    """
    
    # Augment the UV coordinates with a third dimension (W = 1)
    uvw_coords = [(u, v, 1) for u, v in uv_coords]
    
    # Ensure there are at least 4 point correspondences.
    if len(uvw_coords) < 4 or len(xyz_coords) < 4:
        raise ValueError("At least 4 point correspondences are required.")
    
    # Construct the matrix A.
    A = []
    for (u, v, w), (x, y, z) in zip(uvw_coords, xyz_coords):
        A.append([u, v, w, 1, 0, 0, 0, 0, 0, 0, 0, 0, -u*x, -v*x, -w*x, -x])
        A.append([0, 0, 0, 0, u, v, w, 1, 0, 0, 0, 0, -u*y, -v*y, -w*y, -y])
        A.append([0, 0, 0, 0, 0, 0, 0, 0, u, v, w, 1, -u*z, -v*z, -w*z, -z])
        
    A = np.array(A)
    
    # Singular Value Decomposition to solve for the transformation matrix.
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(4, 4)
    print(H)
    return H


def transform_uv_to_real(uv_coords, matrix):
    # Augment the UV coordinates with a third and fourth dimension (W = 1 and homogeneous 1)
    uvw1 = np.hstack((uv_coords, np.ones((uv_coords.shape[0], 1)), np.ones((uv_coords.shape[0], 1))))
    
    # Multiply the UVW1 coordinates by the transformation matrix
    xyzw = np.dot(uvw1, matrix.T)
    
    # Convert back to 3D real-world coordinates
    return xyzw[:, :3] / xyzw[:, 3:4]

def transform_real_to_uv(xyz_coords, matrix):
    """
    Transforms real-world coordinates to UV coordinates using the given matrix.
    
    Parameters:
    - xyz_coords: Nx3 numpy array of 3D real-world coordinates.
    - matrix: 4x4 projective transformation matrix.
    
    Returns:
    - Nx2 numpy array of 2D image coordinates.
    """
    # Compute the inverse of the transformation matrix
    inv_matrix = np.linalg.inv(matrix)
    
    # Augment the XYZ coordinates with a fourth dimension (homogeneous 1)
    xyzw1 = np.hstack((xyz_coords, np.ones((xyz_coords.shape[0], 1))))
    
    # Multiply the XYZW1 coordinates by the inverse transformation matrix
    uvw = np.dot(xyzw1, inv_matrix.T)
    
    # Convert back to 2D image coordinates
    return uvw[:, :2] / uvw[:, 2:3]

def transform_image_corners(image, matrix):
    """
    Transforms the four corner points of the image to real-world coordinates using the given matrix.
    
    Parameters:
    - image: PIL Image object.
    - matrix: 4x4 projective transformation matrix.
    
    Returns:
    - List of four real-world coordinates.
    """
    width, height = image.size
    corner_uv_coords = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    corner_xyz_coords = transform_uv_to_real(corner_uv_coords, matrix)
    return corner_xyz_coords

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
            self.click_timer.start(300)  # 300 ms 

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
        ellipse_item = self.scene().addEllipse(x-3, y-3, 6, 6, QPen(color), QBrush(color))
        return ellipse_item

    def draw_error_line(self, start_point, end_point, color=Qt.red):
        """Draw an error line between two points on the QGraphicsScene."""
        line_item = self.scene().addLine(start_point[0], start_point[1], end_point[0], end_point[1], QPen(color))
        return line_item
class GeoReferencer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_data = {}
        self.image = None
        self.original_image = None
        self.image_path = ""
        self.refpointnames = []
        self.image_coordinates = []
        self.real_world_coordinates = []
        self.mesh = None
        self.editing_uv_for_row = None

        self.config_file = "config.json"  # Name of the config file

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
        self.table.setColumnCount(4)  # Changed from 3 to 4
        self.table.setHorizontalHeaderLabels(["Name", "Pixel Coordinates", "Real-world Coordinates", "RMSE"])  # Added "Name"
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        self.table.cellDoubleClicked.connect(self.on_table_cell_double_clicked)
        self.table.cellChanged.connect(self.on_table_cell_changed)

        # Buttons

        self.calc_transform_button = QPushButton("Calculate Transformation", self)
        self.calc_transform_button.clicked.connect(self.calculate_transformation)
        layout.addWidget(self.calc_transform_button)

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
    
    def create_plane_and_box_with_corners(self, image_points, ref_points, image_path, image):
        ref_points = np.array(ref_points)
        image_points = np.array(image_points)
        T = compute_projective_transformation(image_points, ref_points)
        img = o3d.io.read_image(image_path)
        img_width, img_height = np.asarray(img).shape[1], np.asarray(img).shape[0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ref_points)

        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=100)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal) 
        print(f"Plane normal: {normal}")

        image_corners_xyz = transform_image_corners(image, T)
        corner_pcd = o3d.geometry.PointCloud()
        corner_pcd.points = o3d.utility.Vector3dVector(image_corners_xyz)
        extruded_image_corners = copy.deepcopy(corner_pcd).translate((-self.config_data['frame_thickness']* normal[0], -self.config_data['frame_thickness'] * normal[1], -self.config_data['frame_thickness'] * normal[2]), relative=True)

        normals_for_ref = np.tile(normal, (len(ref_points), 1))
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

        # Identify the two faces that are most parallel to the XY plane
        dot_products = np.dot(face_normals, normal)
        sorted_indices = np.argsort(dot_products)
        target_face_indices = sorted_indices[-2:]  # Get the last two indices (highest dot products)

        print(f"Target face indices: {target_face_indices}")

        # UV coordinates for the entire texture
        full_texture_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        # For simplicity, other vertices will map to a tiny portion of the texture (e.g., top-left corner)
        tiny_texture_uv = np.array([[0, 0]])




        # Gather all unique vertices from the target faces
        unique_vertices = set()
        for target_face_idx in target_face_indices:
            face_vertices = np.asarray(box_mesh.triangles)[target_face_idx]
            unique_vertices.update(face_vertices)

        # Transform and normalize vertices
        img = o3d.io.read_image(image_path)
        img_width, img_height = np.asarray(img).shape[1], np.asarray(img).shape[0]

        # Create a mapping of vertex indices to their UV coordinates
        vertex_to_uv = {}

        for vert_idx, vert in enumerate(box_mesh.vertices):
            if vert_idx in unique_vertices:
                xyz = np.asarray(vert)
                uv = transform_real_to_uv(np.array([xyz]), T)[0]
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
        if not self.real_world_coordinates:
            print("No reference points available.")
            return

        self.create_plane_and_box_with_corners(self.image_coordinates, self.real_world_coordinates, self.image_path, self.original_image)
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

    def on_image_click(self, x, y):
        if hasattr(self, 'editing_uv_for_row') and self.editing_uv_for_row is not None:
            self.image_coordinates[self.editing_uv_for_row] = (x, y)
            self.table.setItem(self.editing_uv_for_row, 1, QTableWidgetItem(f"{x},{y}"))
            delattr(self, 'editing_uv_for_row')

            self.update_reference_and_corner_points_on_canvas()
        else:
            # Store clicked image coordinates
            self.image_coordinates.append((x, y))

            # Prompt user for real-world coordinates
            coords, ok = QInputDialog.getText(self, "Input", "Enter real-world coordinates (x, y, z):")

            if ok and coords:
                coords_list = coords.split(',')
                if len(coords_list) != 3:
                    print("Invalid coordinates.")
                    return
                rw_x, rw_y, rw_z = map(float, coords_list)
                self.real_world_coordinates.append((rw_x, rw_y, rw_z))

                # Draw a point on the image where the user clicked
                draw = ImageDraw.Draw(self.image)
                draw.ellipse([(x-3, y-3), (x+3, y+3)], fill='red')

                # Update the image
                qim = QImage(self.image.tobytes("raw", "RGBA"), self.image.width, self.image.height, QImage.Format_RGBA8888)
                pixmap = QPixmap.fromImage(qim)
                self.scene.clear()
                self.scene.addPixmap(pixmap)
                self.view.setScene(self.scene)

                # Add to the reference table
                row_position = self.table.rowCount()
                self.table.insertRow(row_position)
                self.table.setItem(row_position, 0, QTableWidgetItem(f"{x},{y}"))
                self.table.setItem(row_position, 1, QTableWidgetItem(f"{rw_x},{rw_y},{rw_z}"))

                self.update_reference_and_corner_points_on_canvas()

    def ask_for_image_file(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "JPEG files (*.jpg;*.jpeg);;PNG files (*.png);;All Files (*)", options=options)
        return filepath

    @pyqtSlot()
    def load_image(self):
        ## developer setup
        #self.image_path = self.ask_for_image_file()
        self.image_path = "C:/Users/tronc/Nextcloud/Uruk/WES_paleoenvi/URUK_ERT/Uruk_2023_03_18_Profile_11_SN_Schlumberger_cut.jpeg"
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
        if column == 1:  # Assuming u,v is column 1
            self.editing_uv_for_row = row  # Create this attribute in __init__
            self.view.scene().removeItem(self.image_coordinates[row])  # Assuming you store QGraphicsItem for each point

    def on_table_cell_changed(self, row, column):
        value = self.table.item(row, column).text()

        # If it's the name column, skip processing
        if column == 0:
            return

        # If it's the image_coordinates column
        if column == 1:
            try:
                x, y = map(int, value.split(','))
                self.image_coordinates[row] = (x, y)
            except ValueError:
                # Handle invalid values (either by logging, showing a message, or setting a default value)
                pass

        # If it's the real_world_coordinates column
        if column == 2:
            try:
                x, y, z = map(float, value.split(','))
                self.real_world_coordinates[row] = (x, y, z)
            except ValueError:
                # Handle invalid values (either by logging, showing a message, or setting a default value)
                pass

        self.update_reference_and_corner_points_on_canvas()

    @pyqtSlot()
    def save_reference_points(self):
        if not self.image_coordinates or not self.real_world_coordinates:
            print("No reference points to save.")
            return

        # Preparing the data to be saved
        data = {
            'image_coordinates': self.image_coordinates,
            'real_world_coordinates': self.real_world_coordinates
        }

        # Propose a default filename as the input image name with .json extension
        default_filename = self.image_path.rsplit('.', 1)[0] + '.json'
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Reference Points", default_filename, "JSON files (*.json);;All Files (*)")

        if filepath:
            with open(filepath, 'w') as file:
                json.dump(data, file)
            print(f"Reference points saved to {filepath}")

    @pyqtSlot()
    def load_reference_points(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select a JSON or CSV file", "", "JSON files (*.json);;CSV files (*.csv);;All Files (*)")

        if filepath.endswith(".json"):
            with open(filepath, 'r') as file:
                data = json.load(file)
            self.refpointnames = data['name']
            self.image_coordinates = data['image_coordinates']
            self.real_world_coordinates = data['real_world_coordinates']

        elif filepath.endswith(".csv"):
            import csv
            with open(filepath, 'r') as file:
                reader = csv.reader(file)
                csv_data = list(reader)[:21]  # First 20 rows + header

            dialog = CSVStructureDialog(csv_data)
            result = dialog.exec_()

            if result == QDialog.Accepted:
                self.refpointnames = []
                self.image_coordinates = []
                self.real_world_coordinates = []
                
                for row_data in csv_data[1:]:
                    name = row_data[dialog.mapping["name"]] if "name" in dialog.mapping else ""
                    u, v = (row_data[dialog.mapping["u"]], row_data[dialog.mapping["v"]]) if "u" in dialog.mapping and "v" in dialog.mapping else (None, None)
                    x, y, z = (row_data[dialog.mapping["x"]], row_data[dialog.mapping["y"]], row_data[dialog.mapping["z"]]) if all(key in dialog.mapping for key in ["x", "y", "z"]) else (None, None, None)

                    self.refpointnames.append(name)
                    self.image_coordinates.append((u, v))
                    self.real_world_coordinates.append((x, y, z))

        # Determine the number of rows based on the longest list
        num_rows = max(len(self.refpointnames), len(self.image_coordinates), len(self.real_world_coordinates))
        self.table.setRowCount(num_rows)

        # Populate the table with the data from the lists
        for i in range(num_rows):
            if i < len(self.refpointnames):
                self.table.setItem(i, 0, QTableWidgetItem(self.refpointnames[i]))
            if i < len(self.image_coordinates):
                u, v = self.image_coordinates[i]
                self.table.setItem(i, 1, QTableWidgetItem(f"{u},{v}"))
            if i < len(self.real_world_coordinates):
                x, y, z = self.real_world_coordinates[i]
                self.table.setItem(i, 2, QTableWidgetItem(f"{x},{y},{z}"))

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
        #mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(depth=8)
        #radii = [10, 20, 30, 40]
        #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            #box_pcd_utm, o3d.utility.DoubleVector(radii))
        #mesh.compute_vertex_normals()
        #print('Box points ',np.asarray(box.get_box_points()))
        #boxptcloud = o3d.geometry.PointCloud()
        #boxptcloud.points = o3d.utility.Vector3dVector(np.asarray(box.get_box_points()))
        #tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(box_pcd_utm)
        #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(box_pcd_utm, 0.6)
       # mesh.compute_vertex_normals()
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
        pcd.points = o3d.utility.Vector3dVector(self.real_world_coordinates)

        # Check if the PointCloud has normals, if not compute them
        if not pcd.has_normals():
            pcd.estimate_normals()

        # Visualization objects
        vis_objects = [pcd]

        # If a mesh exists, add it to the visualization
        if hasattr(self, 'mesh') and self.mesh.vertices:
            vis_objects.append(self.mesh)

        o3d.visualization.draw_geometries(vis_objects)
    
    def update_reference_and_corner_points_on_canvas(self):
        """
        Update the reference and corner points on the canvas.
        """
        # Clear the previous vector items from the scene
        for item in self.view.scene().items():
            if isinstance(item, (QGraphicsEllipseItem, QGraphicsLineItem)):
                self.view.scene().removeItem(item)

        # Get transformed UV coordinates using the transformation matrix
        if hasattr(self, 'transformation_matrix'):  # Ensure transformation matrix is available
            transformed_uv = transform_real_to_uv(np.array(self.real_world_coordinates), self.transformation_matrix)
            corners_uv = transform_real_to_uv(transform_image_corners(self.original_image, self.transformation_matrix), self.transformation_matrix)
        else:
            transformed_uv = self.image_coordinates  # Default to original if no transformation matrix available
            corners_uv = [(0, 0), (self.original_image.width, 0), (0, self.original_image.height), (self.original_image.width, self.original_image.height)]

        # Draw points
        for orig, trans in zip(self.image_coordinates, transformed_uv):
            # Original user-input points
            self.view.draw_reference_point(orig[0], orig[1], color=Qt.red)
            # Transformed points
            self.view.draw_reference_point(trans[0], trans[1], color=Qt.blue)
            # Error line
            self.view.draw_error_line(orig, trans, color=Qt.red)

        # Draw corner points in green
        for corner in corners_uv:
            self.view.draw_reference_point(corner[0], corner[1], color=Qt.green)


    def calculate_transformation(self):
        if len(self.image_coordinates) < 4 or len(self.real_world_coordinates) < 4:
            print("At least 4 reference points are required.")
            return
        
        # Compute the transformation matrix
        self.transformation_matrix = compute_projective_transformation(np.array(self.image_coordinates), np.array(self.real_world_coordinates))
        print('Check: ',self.image_coordinates, self.transformation_matrix)
        # Compute transformed UV coordinates
        transformed_real_coords = transform_uv_to_real(np.array(self.image_coordinates), self.transformation_matrix)
        
        # Calculate RMSE values for each point and populate the table
        for i, (trans, real) in enumerate(zip(transformed_real_coords, self.real_world_coordinates)):
            rmse = np.sqrt(np.sum((np.array(trans) - np.array(real))**2))
            self.table.setItem(i, 2, QTableWidgetItem(f"{rmse:.4f}"))
        

class CSVStructureDialog(QDialog):
    def __init__(self, csv_data):
        super().__init__()
        
        self.csv_data = csv_data
        self.mapping = {}

        layout = QVBoxLayout(self)

        # Create the table
        self.table = QTableWidget(self)
        self.table.setColumnCount(len(csv_data[0]))
        self.table.setHorizontalHeaderLabels(csv_data[0])  # Using the first row as headers
        
        for row_data in csv_data[1:]:  # Skip the header
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            for col, item in enumerate(row_data):
                self.table.setItem(row_position, col, QTableWidgetItem(item))
        
        layout.addWidget(self.table)

        # Dropdown menu for mapping columns
        mapping_layout = QHBoxLayout()
        for col in range(self.table.columnCount()):
            combo = QComboBox(self)
            combo.addItems(["-", "u", "v", "x", "y", "z", "name"])  # Added "name" here
            combo.currentIndexChanged.connect(lambda idx, col=col, combo=combo: self.update_mapping(col, combo))
            combo.setFixedWidth(self.table.columnWidth(col))  # Set the width of the combo box to match the column width
            mapping_layout.addWidget(combo)
        layout.addLayout(mapping_layout)

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
        self.accept()