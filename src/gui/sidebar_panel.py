"""
Sidebar panel with display controls and obstacle editing.
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QComboBox, QCheckBox,
    QLabel, QLineEdit, QPushButton, QTreeWidget, QTreeWidgetItem,
    QScrollArea, QMessageBox, QDialog,
)

from config import SimConfig, ObstacleDef, PlacedObstacle, PRECONFIGURED_OBSTACLES


class ObstacleSelector(QWidget):
    obstacle_added = pyqtSignal(ObstacleDef, float, float)

    def __init__(self, obs: ObstacleDef, parent=None):
        super().__init__(parent)
        self.obs = obs

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(obs.name))

        self.x_edit = QLineEdit()
        self.x_edit.setPlaceholderText("X")
        self.x_edit.setMinimumHeight(30)
        layout.addWidget(self.x_edit)

        self.y_edit = QLineEdit()
        self.y_edit.setPlaceholderText("Y")
        self.y_edit.setMinimumHeight(30)
        layout.addWidget(self.y_edit)

        add_button = QPushButton("Add Obstacle")
        add_button.setMinimumHeight(32)
        add_button.clicked.connect(self._add_clicked)
        layout.addWidget(add_button)

    def _add_clicked(self):
        try:
            x = float(self.x_edit.text())
            y = float(self.y_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "X and Y must be numbers.")
            return
        self.obstacle_added.emit(self.obs, x, y)


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
    scalar_field_changed = pyqtSignal(str)
    layer_toggled = pyqtSignal(str, bool)
    obstacles_changed = pyqtSignal()
    obstacle_added = pyqtSignal(ObstacleDef, float, float)

    def __init__(self, config: SimConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.placed_obstacles: list[PlacedObstacle] = []
        self._next_obstacle_id = 0

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        layout.addWidget(QLabel("Scalar Field"))
        self.scalar_combo = QComboBox()
        self.scalar_combo.addItems(["Height", "Velocity", "Pressure", "LAVD"])
        self.scalar_combo.currentTextChanged.connect(self.scalar_field_changed.emit)
        layout.addWidget(self.scalar_combo)

        self.layers = ["water surface", "particles", "particle trails", "vorticity"]
        self.layer_checks = []
        for layer in self.layers:
            checkbox = QCheckBox(layer)
            checkbox.setChecked(True)
            checkbox.toggled.connect(lambda checked, name=layer: self.layer_toggled.emit(name, checked))
            self.layer_checks.append(checkbox)
            layout.addWidget(checkbox)

        layout.addWidget(QLabel("Obstacles"))
        self.obstacle_tree = QTreeWidget()
        self.obstacle_tree.setColumnCount(3)
        self.obstacle_tree.setHeaderLabels(["Type", "X", "Y"])
        self.obstacle_tree.itemDoubleClicked.connect(self._handle_obstacle_double_click)
        layout.addWidget(self.obstacle_tree)

        layout.addWidget(QLabel("Add Obstacle"))
        obstacle_panel_widget = QWidget()
        obstacle_panel = QVBoxLayout(obstacle_panel_widget)
        for od in PRECONFIGURED_OBSTACLES:
            box = QGroupBox()
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

        item = QTreeWidgetItem([obs.name, f"{x:.2f}", f"{y:.2f}"])
        item.setData(0, Qt.UserRole, placed_obs.obstacle_id)
        self.obstacle_tree.addTopLevelItem(item)

        self.obstacle_added.emit(obs, x, y)
        self.obstacles_changed.emit()

    def _handle_obstacle_double_click(self, item, _column):
        dialog = DeleteObstacleDialog(self)
        if not dialog.exec():
            return

        obstacle_id = item.data(0, Qt.UserRole)
        self.placed_obstacles = [
            obs for obs in self.placed_obstacles
            if obs.obstacle_id != obstacle_id
        ]
        index = self.obstacle_tree.indexOfTopLevelItem(item)
        if index >= 0:
            self.obstacle_tree.takeTopLevelItem(index)
        self.obstacles_changed.emit()