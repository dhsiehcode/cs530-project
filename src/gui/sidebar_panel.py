"""
Sidebar panel with three sections:
  1. Display  – scalar-field selector, layer toggles
  2. Obstacle Editor – tree of placed obstacles, add / remove
  3. Simulation  – CPU/GPU selector, Run button, info bar
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QComboBox, QCheckBox,
    QLabel, QLineEdit, QPushButton, QTreeWidget, QTreeWidgetItem,
    QHBoxLayout, QProgressBar, QScrollArea, QMessageBox, QDialog,
    QDialogButtonBox, QFormLayout, QStackedWidget,
)

from config import SimConfig, ObstacleDef, PlacedObstacle, PRECONFIGURED_OBSTACLES


## helps add obstacle
class ObstacleSelector(QWidget):
    obstacle_added = pyqtSignal(ObstacleDef, float, float)

    def __init__(self, obs : ObstacleDef, parent=None):
        super().__init__(parent)
        self.name = obs.name
        self.obs = obs
        layout = QVBoxLayout(self)
        #layout.setContentsMargins(6, 6, 6, 6)

        layout.addWidget(QLabel(f"{self.name}"))

        self.x_edit = QLineEdit()
        self.x_edit.setPlaceholderText("X")
        self.x_edit.setMinimumHeight(30)

        self.y_edit = QLineEdit()
        self.y_edit.setPlaceholderText("Y")
        self.y_edit.setMinimumHeight(30)

        layout.addWidget(self.x_edit)
        layout.addWidget(self.y_edit)
        layout.addSpacing(6)

        add_button = QPushButton("Add Obstacle")
        add_button.setMinimumHeight(32)
        layout.addWidget(add_button)

        add_button.clicked.connect(self._add_clicked)

    def _add_clicked(self):
        try:
            x = float(self.x_edit.text())
            y = float(self.y_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "X and Y must be numbers.")
            return

        self.obstacle_added.emit(self.obs, x, y)

## help delete obstacle
class DeleteObstacleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Delete Obstacle")
        self.setModal(True)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Delete this obstacle?"))

        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self.accept)

        layout.addWidget(delete_button)


class SidebarPanel(QWidget):
    """Right-hand sidebar with display options, obstacle editor, sim controls."""


    scalar_field_changed = pyqtSignal(str) # "height", "speed" or "vorticity"
    layer_toggled = pyqtSignal(str, bool)  # ["water surface", "glyphs", "streamlines"]    
    obstacles_changed = pyqtSignal() # change of obstacle
    obstacle_added = pyqtSignal(ObstacleDef, float, float)

    def __init__(self, config: SimConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.placed_obstacles: list = []            # list[PlacedObstacle]
        self._next_obstacle_id = 0

        ## laytout
        layout = QVBoxLayout(self)
        #layout.setSpacing(8)
        #layout.setContentsMargins(8, 8, 8, 8)
        layout.setAlignment(Qt.AlignTop)

        ## scalar field selector
        scalar_label = QLabel("Scalar Field")
        self.scalar_combo = QComboBox()
        self.scalar_combo.addItems([
                    "Height",
                    "Speed",
                    "Vorticity"
        ])
        layout.addWidget(scalar_label)
        layout.addWidget(self.scalar_combo)
        self.scalar_combo.currentTextChanged.connect(
            self.scalar_field_changed.emit
        )

        ## layers selector
        self.layers = ["water surface", "glyphs", "streamlines","contours"]
        self.layer_checks = []
        for layer in self.layers:
            checkbox = QCheckBox(layer)
            checkbox.setChecked(True)
            checkbox.toggled.connect(
                lambda checked, l=layer: self.layer_toggled.emit(l, checked)
            )
            self.layer_checks.append(checkbox)
            layout.addWidget(checkbox)
        layout.addStretch()

        ## obstacle  display
        layout.addWidget(QLabel("Obstacles"))
        self.obstacle_tree = QTreeWidget()
        self.obstacle_tree.setColumnCount(3)
        self.obstacle_tree.setHeaderLabels(["Type", "X", "Y"])
        self.obstacle_tree.itemDoubleClicked.connect(self._handle_obstacle_double_click)
        layout.addWidget(self.obstacle_tree)

        ## obstacle selector (scrollable)
        layout.addWidget(QLabel("Add Obstacle"))
        obstacle_panel_widget = QWidget()
        obstacle_panel = QVBoxLayout(obstacle_panel_widget)
        for od in PRECONFIGURED_OBSTACLES:
            box = QGroupBox()
            obstacle_font = box.font()
            obstacle_font.setPointSize(obstacle_font.pointSize() + 2)
            box.setFont(obstacle_font)
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(0, 0, 0, 0)
            selector = ObstacleSelector(od)
            selector.obstacle_added.connect(self._add_obstacle)
            box_layout.addWidget(selector)
            obstacle_panel.addWidget(box)
        obstacle_panel.addStretch()

        obstacle_scroll = QScrollArea()
        obstacle_scroll.setWidgetResizable(True)
        obstacle_scroll.setWidget(obstacle_panel_widget)
        layout.addWidget(obstacle_scroll)
        layout.addStretch()

    def current_scalar_field(self) -> str:
            return self.scalar_combo.currentText()

    def is_layer_enabled(self, index: int) -> bool:
            if 0 <= index < len(self.layer_checks):
                return self.layer_checks[index].isChecked()
            return False
    
    def _add_obstacle(self, obs: ObstacleDef, x: float, y: float):
        placed_obs = PlacedObstacle(obs, self._next_obstacle_id, x, y)
        self._next_obstacle_id += 1
        self.placed_obstacles.append(placed_obs)
        item = QTreeWidgetItem([
            obs.name,
            f"{x:.2f}",
            f"{y:.2f}"
        ])
        item.setData(0, Qt.UserRole, placed_obs.obstacle_id)
        self.obstacle_tree.addTopLevelItem(item)
        self.obstacle_added.emit(obs, x, y)
        self.obstacles_changed.emit()

    def _handle_obstacle_double_click(self, item, column):
        dialog = DeleteObstacleDialog(self)

        if dialog.exec():
            obstacle_id = item.data(0, Qt.UserRole)

            # Remove from obstacle list
            self.placed_obstacles = [
                obs for obs in self.placed_obstacles
                if obs.obstacle_id != obstacle_id
            ]
            index = self.obstacle_tree.indexOfTopLevelItem(item)
            if index >= 0:
                self.obstacle_tree.takeTopLevelItem(index)
            self.obstacles_changed.emit()




