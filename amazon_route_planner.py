import sys
import math
import heapq
from collections import deque

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

FONT_FAMILIES = ["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"]
MONO_FAMILIES = "'Cascadia Code', 'Consolas', 'Menlo', monospace"

plt.rcParams["font.family"] = FONT_FAMILIES

LIGHT_THEME = dict(
    window_bg="#f6f4ef",
    panel_bg="#efebe3",
    surface="#ffffff",
    border="#d9d2c3",
    hover_surface="#e6e0d3",
    text="#1e293b",
    text_muted="#64748b",
    accent="#c9a227",
    accent_hover="#b38f28",
    accent_text="#1a1405",
    toggle_bg="#1f2937",
    toggle_hover="#111827",
    toggle_text="#f9fafb",
    map_bg="#faf8f5",
    map_axes="#ffffff",
    grid="#e2d8c3",
    node="#d4af37",
    node_edge="#b38f28",
    start="#10b981",
    start_edge="#047857",
    goal="#e11d48",
    goal_edge="#be123c",
    on_path="#b8860b",
    on_path_edge="#8b6508",
    edge="#cbd5e1",
    route="#b8860b",
    label_bg="#ffffff",
    label_edge="#d4af37",
    title="#8b6508",
    subtitle="#64748b",
)

DARK_THEME = dict(
    window_bg="#101215",
    panel_bg="#171a1f",
    surface="#1f232a",
    border="#303641",
    hover_surface="#262b33",
    text="#e5e7eb",
    text_muted="#9ca3af",
    accent="#d4af37",
    accent_hover="#e2c25a",
    accent_text="#1a1405",
    toggle_bg="#e5e7eb",
    toggle_hover="#ffffff",
    toggle_text="#111827",
    map_bg="#14161a",
    map_axes="#1b1e24",
    grid="#2a2e36",
    node="#d4af37",
    node_edge="#f0d36b",
    start="#34d399",
    start_edge="#10b981",
    goal="#fb7185",
    goal_edge="#e11d48",
    on_path="#f0b429",
    on_path_edge="#ffd666",
    edge="#3b4252",
    route="#f0b429",
    label_bg="#23272f",
    label_edge="#5b4d1c",
    title="#f0d36b",
    subtitle="#9ca3af",
)

ALGORITHM_PROPERTIES = {
    "UCS": {"complete": "Yes", "optimal": "Yes"},
    "A* Search": {"complete": "Yes", "optimal": "Yes (admissible heuristic)"},
    "GBFS": {"complete": "Yes (finite graph only)", "optimal": "No"},
    "BFS": {"complete": "Yes", "optimal": "Yes, by hop count only (not by distance)"},
    "DFS": {"complete": "Yes (finite graph only)", "optimal": "No"},
}

DEPARTMENT_LINKS = [
    ("Electronics", "Video Games", 440),
    ("Electronics", "Office Products", 500),
    ("Electronics", "Toys & Games", 520),
    ("Video Games", "Movies & Music", 170),
    ("Movies & Music", "Books", 110),
    ("Books", "Toys & Games", 185),
    ("Toys & Games", "Baby", 175),
    ("Baby", "Health & Household", 190),
    ("Baby", "Fashion", 100),
    ("Health & Household", "Beauty", 590),
    ("Health & Household", "Grocery", 640),
    ("Beauty", "Jewelry", 100),
    ("Jewelry", "Fashion", 110),
    ("Fashion", "Shoes", 540),
    ("Shoes", "Sports & Outdoors", 320),
    ("Sports & Outdoors", "Automotive", 250),
    ("Sports & Outdoors", "Garden & Outdoor", 210),
    ("Automotive", "Tools & Home Improvement", 130),
    ("Tools & Home Improvement", "Garden & Outdoor", 100),
    ("Garden & Outdoor", "Furniture", 140),
    ("Furniture", "Home & Kitchen", 110),
    ("Home & Kitchen", "Grocery", 120),
    ("Grocery", "Pet Supplies", 100),
    ("Pet Supplies", "Office Products", 130),
]


def build_stylesheet(theme):
    return f"""
        QMainWindow, QMessageBox {{
            background-color: {theme['window_bg']};
        }}
        QWidget {{
            font-family: {', '.join(f'"{name}"' for name in FONT_FAMILIES[:3])};
            color: {theme['text']};
        }}
        QWidget#sidePanel {{
            background-color: {theme['panel_bg']};
            border-right: 1px solid {theme['border']};
        }}
        QLabel {{
            background: transparent;
        }}
        QLabel#panelTitle {{
            color: {theme['title']};
            font-size: 17pt;
            font-weight: 700;
        }}
        QLabel#panelSubtitle {{
            color: {theme['text_muted']};
            font-size: 9pt;
            padding-bottom: 10px;
        }}
        QLabel#sectionLabel {{
            color: {theme['text_muted']};
            font-size: 9pt;
            font-weight: 600;
            padding-top: 6px;
        }}
        QComboBox {{
            background-color: {theme['surface']};
            border: 1px solid {theme['border']};
            border-radius: 6px;
            padding: 7px 10px;
            min-height: 20px;
        }}
        QComboBox:hover {{
            border-color: {theme['accent']};
        }}
        QComboBox:disabled {{
            color: {theme['text_muted']};
        }}
        QComboBox QAbstractItemView {{
            background-color: {theme['surface']};
            border: 1px solid {theme['border']};
            selection-background-color: {theme['accent']};
            selection-color: {theme['accent_text']};
            outline: none;
        }}
        QPushButton {{
            border-radius: 6px;
            padding: 10px 14px;
            font-weight: 600;
            border: 1px solid transparent;
        }}
        QPushButton#primaryButton {{
            background-color: {theme['accent']};
            color: {theme['accent_text']};
        }}
        QPushButton#primaryButton:hover {{
            background-color: {theme['accent_hover']};
        }}
        QPushButton#secondaryButton {{
            background-color: transparent;
            border: 1px solid {theme['border']};
            color: {theme['text']};
        }}
        QPushButton#secondaryButton:hover {{
            background-color: {theme['hover_surface']};
        }}
        QPushButton#themeButton {{
            background-color: {theme['toggle_bg']};
            color: {theme['toggle_text']};
        }}
        QPushButton#themeButton:hover {{
            background-color: {theme['toggle_hover']};
        }}
        QPushButton:disabled {{
            background-color: {theme['hover_surface']};
            color: {theme['text_muted']};
            border: 1px solid {theme['border']};
        }}
        QTextEdit#results {{
            background-color: {theme['surface']};
            border: 1px solid {theme['border']};
            border-radius: 6px;
            padding: 8px;
            font-family: {MONO_FAMILIES};
            font-size: 10pt;
        }}
        QStatusBar {{
            background-color: {theme['panel_bg']};
            color: {theme['text_muted']};
            border-top: 1px solid {theme['border']};
        }}
        QToolBar {{
            background-color: {theme['window_bg']};
            border: none;
            spacing: 4px;
        }}
        QToolButton:hover {{
            background-color: {theme['hover_surface']};
            border-radius: 4px;
        }}
        QToolTip {{
            background-color: {theme['surface']};
            color: {theme['text']};
            border: 1px solid {theme['border']};
        }}
    """


def build_palette(theme):
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(theme["window_bg"]))
    palette.setColor(QPalette.WindowText, QColor(theme["text"]))
    palette.setColor(QPalette.Base, QColor(theme["surface"]))
    palette.setColor(QPalette.AlternateBase, QColor(theme["panel_bg"]))
    palette.setColor(QPalette.Text, QColor(theme["text"]))
    palette.setColor(QPalette.Button, QColor(theme["surface"]))
    palette.setColor(QPalette.ButtonText, QColor(theme["text"]))
    palette.setColor(QPalette.ToolTipBase, QColor(theme["surface"]))
    palette.setColor(QPalette.ToolTipText, QColor(theme["text"]))
    palette.setColor(QPalette.Highlight, QColor(theme["accent"]))
    palette.setColor(QPalette.HighlightedText, QColor(theme["accent_text"]))
    return palette


class RouteNavigatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Amazon Shop Departments Navigation System")
        self.setGeometry(100, 100, 1350, 820)

        self.graph = nx.Graph()
        self._load_departments()
        self.positions = nx.spring_layout(self.graph, seed=7, k=2.5)

        self.dark_mode = False
        self.theme = LIGHT_THEME
        self.last_drawn = {"path": None, "start": None, "goal": None}

        self.animation_path = []
        self.animation_index = 0
        self.animation_info = {}
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._advance_animation)

        self._build_interface()
        self._apply_theme()
        self.status_bar.showMessage("Ready. Select a start, goal, and algorithm, then click Find Route.")

    def _load_departments(self):
        for first, second, distance in DEPARTMENT_LINKS:
            self.graph.add_edge(first, second, weight=distance)

    def _build_interface(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_side_panel())
        root_layout.addWidget(self._build_map_area(), 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _build_side_panel(self):
        panel = QWidget()
        panel.setObjectName("sidePanel")
        panel.setFixedWidth(360)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(8)

        title = QLabel("Route Planner")
        title.setObjectName("panelTitle")
        subtitle = QLabel("Shop department navigation")
        subtitle.setObjectName("panelSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        department_names = sorted(self.graph.nodes())

        layout.addWidget(self._section_label("Start location"))
        self.start_combo = QComboBox()
        self.start_combo.addItems(department_names)
        layout.addWidget(self.start_combo)

        layout.addWidget(self._section_label("Goal location"))
        self.goal_combo = QComboBox()
        self.goal_combo.addItems(department_names)
        if self.goal_combo.count() > 1:
            self.goal_combo.setCurrentIndex(1)
        layout.addWidget(self.goal_combo)

        layout.addWidget(self._section_label("Algorithm"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["A* Search", "UCS", "GBFS", "BFS", "DFS"])
        layout.addWidget(self.algo_combo)

        layout.addSpacing(8)
        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        self.run_button = self._make_button("Find Route", "primaryButton", self.run_search)
        self.reset_button = self._make_button("Reset", "secondaryButton", self.reset_view)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.reset_button)
        layout.addLayout(button_row)

        self.theme_button = self._make_button("Dark Mode", "themeButton", self.toggle_theme)
        layout.addWidget(self.theme_button)

        layout.addSpacing(8)
        layout.addWidget(self._section_label("Route details"))
        self.results_text = QTextEdit()
        self.results_text.setObjectName("results")
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text, 1)

        return panel

    def _build_map_area(self):
        container = QWidget()
        self.map_layout = QVBoxLayout(container)
        self.map_layout.setContentsMargins(8, 8, 8, 8)

        self.figure, self.axes = plt.subplots(figsize=(9, 9))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.map_layout.addWidget(self.toolbar)
        self.map_layout.addWidget(self.canvas)
        return container

    def _section_label(self, text):
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        return label

    def _make_button(self, text, object_name, handler):
        button = QPushButton(text)
        button.setObjectName(object_name)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(handler)
        return button

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self._apply_theme()

    def _apply_theme(self):
        self.theme = DARK_THEME if self.dark_mode else LIGHT_THEME
        QApplication.setPalette(build_palette(self.theme))
        self.setStyleSheet(build_stylesheet(self.theme))
        self.theme_button.setText("Light Mode" if self.dark_mode else "Dark Mode")
        self._rebuild_toolbar()
        self.draw_graph(**self.last_drawn)

    def _rebuild_toolbar(self):
        fresh_toolbar = NavigationToolbar(self.canvas, self)
        self.map_layout.replaceWidget(self.toolbar, fresh_toolbar)
        self.toolbar.deleteLater()
        self.toolbar = fresh_toolbar

    def heuristic(self, node_a, node_b):
        x1, y1 = self.positions[node_a]
        x2, y2 = self.positions[node_b]
        return math.hypot(x2 - x1, y2 - y1)

    def path_weight(self, path):
        return sum(self.graph[a][b]["weight"] for a, b in zip(path[:-1], path[1:]))

    def _rebuild_path(self, came_from, goal):
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path

    def ucs(self, start, goal):
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        visited = []

        while frontier:
            cost, current = heapq.heappop(frontier)
            if current in visited:
                continue
            visited.append(current)
            if current == goal:
                break

            for neighbor in self.graph.neighbors(current):
                new_cost = cost + self.graph[current][neighbor]["weight"]
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(frontier, (new_cost, neighbor))

        if goal not in came_from:
            raise nx.NetworkXNoPath(f"No path found by UCS from {start} to {goal}")
        return self._rebuild_path(came_from, goal), visited, cost_so_far[goal]

    def astar(self, start, goal):
        frontier = [(self.heuristic(start, goal), start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        visited = []

        while frontier:
            _, current = heapq.heappop(frontier)
            if current in visited:
                continue
            visited.append(current)
            if current == goal:
                break

            for neighbor in self.graph.neighbors(current):
                new_cost = cost_so_far[current] + self.graph[current][neighbor]["weight"]
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    came_from[neighbor] = current
                    heapq.heappush(frontier, (priority, neighbor))

        if goal not in came_from:
            raise nx.NetworkXNoPath(f"No path found by A* from {start} to {goal}")
        return self._rebuild_path(came_from, goal), visited, cost_so_far[goal]

    def gbfs(self, start, goal):
        frontier = [(self.heuristic(start, goal), start)]
        came_from = {start: None}
        visited = []

        while frontier:
            _, current = heapq.heappop(frontier)
            if current in visited:
                continue
            visited.append(current)
            if current == goal:
                break

            for neighbor in self.graph.neighbors(current):
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    heapq.heappush(frontier, (self.heuristic(neighbor, goal), neighbor))

        if goal not in came_from:
            raise nx.NetworkXNoPath(f"No path found by GBFS from {start} to {goal}")
        return self._rebuild_path(came_from, goal), visited

    def bfs(self, start, goal):
        frontier = deque([start])
        came_from = {start: None}
        seen = {start}
        visited = []

        while frontier:
            current = frontier.popleft()
            visited.append(current)
            if current == goal:
                break

            for neighbor in self.graph.neighbors(current):
                if neighbor not in seen:
                    seen.add(neighbor)
                    came_from[neighbor] = current
                    frontier.append(neighbor)

        if goal not in came_from:
            raise nx.NetworkXNoPath(f"No path found by BFS from {start} to {goal}")
        return self._rebuild_path(came_from, goal), visited

    def dfs(self, start, goal):
        frontier = [start]
        came_from = {start: None}
        seen = {start}
        visited = []

        while frontier:
            current = frontier.pop()
            visited.append(current)
            if current == goal:
                break

            for neighbor in self.graph.neighbors(current):
                if neighbor not in seen:
                    seen.add(neighbor)
                    came_from[neighbor] = current
                    frontier.append(neighbor)

        if goal not in came_from:
            raise nx.NetworkXNoPath(f"No path found by DFS from {start} to {goal}")
        return self._rebuild_path(came_from, goal), visited

    def find_route(self, algorithm, start, goal):
        if algorithm == "UCS":
            path, visited, cost = self.ucs(start, goal)
            return path, visited, f"{cost} meters"
        if algorithm == "A* Search":
            path, visited, cost = self.astar(start, goal)
            return path, visited, f"{cost} meters"
        if algorithm == "GBFS":
            path, visited = self.gbfs(start, goal)
            return path, visited, f"{self.path_weight(path)} meters (not guaranteed optimal)"
        if algorithm == "BFS":
            path, visited = self.bfs(start, goal)
            return path, visited, f"{len(path) - 1} hop(s) (unweighted)"
        if algorithm == "DFS":
            path, visited = self.dfs(start, goal)
            return path, visited, f"{len(path) - 1} hop(s) via DFS discovery order"
        raise ValueError(f"Unknown algorithm: {algorithm}")

    def run_search(self):
        start = self.start_combo.currentText()
        goal = self.goal_combo.currentText()
        algorithm = self.algo_combo.currentText()

        if start == goal:
            QMessageBox.warning(self, "Invalid Selection", "Start and goal locations cannot be the same.")
            return

        try:
            path, visited, cost_label = self.find_route(algorithm, start, goal)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as error:
            message = f"No path found between {start} and {goal}.\n({error})"
            self.results_text.setText(message)
            self.status_bar.showMessage("No path found.")
            QMessageBox.critical(self, "Route Error", message)
            return
        except Exception as error:
            message = f"Unexpected error: {error}"
            self.results_text.setText(message)
            self.status_bar.showMessage("Something went wrong.")
            QMessageBox.critical(self, "Error", message)
            return

        self.animation_path = path
        self.animation_index = 0
        self.animation_info = dict(
            algorithm=algorithm, start=start, goal=goal, cost_label=cost_label, visited=visited
        )

        self._set_controls_enabled(False)
        self.results_text.setText(f"Algorithm : {algorithm}\nAnimating route...\n")
        self.status_bar.showMessage(f"Step 1/{len(path)}: at {path[0]}")
        self.draw_graph(path=path[:1], start=start, goal=goal)

        interval = max(180, min(600, 1800 // max(1, len(path))))
        self.animation_timer.start(interval)

    def _advance_animation(self):
        self.animation_index += 1
        partial_path = self.animation_path[: self.animation_index + 1]
        info = self.animation_info
        self.draw_graph(path=partial_path, start=info["start"], goal=info["goal"])

        if self.animation_index >= len(self.animation_path) - 1:
            self._finish_animation()
            return

        current_node = self.animation_path[self.animation_index]
        self.status_bar.showMessage(
            f"Step {self.animation_index + 1}/{len(self.animation_path)}: at {current_node}"
        )
        self.results_text.setText(
            f"Algorithm : {info['algorithm']}\nAnimating: {' -> '.join(partial_path)} ...\n"
        )

    def _finish_animation(self):
        self.animation_timer.stop()
        self._set_controls_enabled(True)

        info = self.animation_info
        properties = ALGORITHM_PROPERTIES[info["algorithm"]]
        path_matches_visited = self.animation_path == info["visited"]

        self.results_text.setText(
            f"Algorithm      : {info['algorithm']}\n"
            f"Path           : {' -> '.join(self.animation_path)}\n"
            f"Visited        : {' -> '.join(info['visited'])}\n"
            f"{'-' * 40}\n"
            f"Path == Visited: {'Yes' if path_matches_visited else 'No'}\n"
            f"Complete       : {properties['complete']}\n"
            f"Optimal        : {properties['optimal']}\n"
            f"Cost           : {info['cost_label']}\n\n"
            f"Status: Path calculated successfully."
        )
        self.status_bar.showMessage(f"{info['algorithm']}: {info['start']} -> {info['goal']} found.")

    def reset_view(self):
        self.animation_timer.stop()
        self._set_controls_enabled(True)
        self.results_text.clear()
        self.last_drawn = {"path": None, "start": None, "goal": None}
        self.draw_graph()
        self.status_bar.showMessage("View reset.")

    def _set_controls_enabled(self, enabled):
        controls = (
            self.run_button,
            self.reset_button,
            self.theme_button,
            self.start_combo,
            self.goal_combo,
            self.algo_combo,
        )
        for control in controls:
            control.setEnabled(enabled)

    def draw_graph(self, path=None, start=None, goal=None):
        self.last_drawn = {"path": path, "start": start, "goal": goal}

        self.axes.clear()
        self._style_axes()
        self._draw_edges(path)
        self._draw_route(path)
        self._draw_nodes(path, start, goal)
        self._draw_legend()
        self._draw_titles()
        self.canvas.draw()

    def _style_axes(self):
        colors = self.theme
        self.figure.patch.set_facecolor(colors["map_bg"])
        self.axes.set_facecolor(colors["map_axes"])

        x_values = [x for x, _ in self.positions.values()]
        y_values = [y for _, y in self.positions.values()]
        margin = 0.3
        self.axes.set_xlim(min(x_values) - margin, max(x_values) + margin)
        self.axes.set_ylim(min(y_values) - margin, max(y_values) + margin)

        self.axes.set_aspect("equal")
        self.axes.grid(True, color=colors["grid"], linestyle=":", linewidth=0.8, zorder=0)
        for spine in self.axes.spines.values():
            spine.set_visible(False)
        self.axes.set_xticks([])
        self.axes.set_yticks([])

    def _draw_edges(self, path):
        colors = self.theme
        route_edges = set()
        if path and len(path) > 1:
            route_edges = set(zip(path[:-1], path[1:])) | set(zip(path[1:], path[:-1]))

        distances = nx.get_edge_attributes(self.graph, "weight")

        for first, second in self.graph.edges():
            x1, y1 = self.positions[first]
            x2, y2 = self.positions[second]

            if (first, second) not in route_edges:
                self.axes.plot(
                    [x1, x2], [y1, y2],
                    color=colors["edge"], linewidth=1.5, zorder=1, solid_capstyle="round",
                )

            distance = distances.get((first, second), distances.get((second, first), ""))
            self.axes.text(
                (x1 + x2) / 2, (y1 + y2) / 2, str(distance),
                fontsize=6.5, ha="center", va="center", color=colors["subtitle"], zorder=2,
                bbox=dict(
                    boxstyle="round,pad=0.15", facecolor=colors["label_bg"],
                    edgecolor=colors["label_edge"], linewidth=0.5, alpha=0.9,
                ),
            )

    def _draw_route(self, path):
        if not path or len(path) < 2:
            return
        for first, second in zip(path[:-1], path[1:]):
            arrow = FancyArrowPatch(
                self.positions[first], self.positions[second],
                connectionstyle="arc3,rad=0.08", arrowstyle="-|>", mutation_scale=12,
                color=self.theme["route"], linewidth=2.5, zorder=3, shrinkA=12, shrinkB=12,
            )
            self.axes.add_patch(arrow)

    def _draw_nodes(self, path, start, goal):
        colors = self.theme
        for node, (x, y) in self.positions.items():
            if node == start:
                face, edge = colors["start"], colors["start_edge"]
            elif node == goal:
                face, edge = colors["goal"], colors["goal_edge"]
            elif path and node in path:
                face, edge = colors["on_path"], colors["on_path_edge"]
            else:
                face, edge = colors["node"], colors["node_edge"]

            self.axes.scatter([x], [y], s=180, facecolor=face, edgecolor=edge, linewidth=1.2, zorder=4)
            self.axes.annotate(
                node, (x, y), xytext=(0, -9), textcoords="offset points",
                ha="center", va="top", fontsize=6.5, fontweight="bold",
                color=colors["text"], zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=colors["label_bg"], edgecolor="none", alpha=0.85),
            )

    def _draw_legend(self):
        colors = self.theme
        entries = [
            ("Start", colors["start"]),
            ("Goal", colors["goal"]),
            ("On Route", colors["on_path"]),
            ("Department", colors["node"]),
        ]
        handles = [
            Line2D([0], [0], marker="s", color="none", label=label, markerfacecolor=color, markersize=10)
            for label, color in entries
        ]
        legend = self.axes.legend(
            handles=handles, loc="upper left", fontsize=7, framealpha=0.95,
            facecolor=colors["label_bg"], edgecolor=colors["label_edge"], labelcolor=colors["text"],
        )
        legend.set_zorder(5)

    def _draw_titles(self):
        colors = self.theme
        self.axes.set_title(
            "Amazon Shop Departments Map",
            fontsize=12, fontweight="bold", color=colors["title"], pad=10,
        )
        self.axes.text(
            0.5, 1.02, "Interactive Route Visualization",
            transform=self.axes.transAxes, ha="center", fontsize=8,
            color=colors["subtitle"], style="italic",
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont(FONT_FAMILIES[0], 10))

    window = RouteNavigatorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
