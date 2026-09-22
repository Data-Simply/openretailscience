"""Grid-based tree diagrams for hierarchical KPI decomposition.

Internal: TreeGrid lays out pluggable TreeNode subclasses on a grid and draws
elbow connectors between parent and child nodes.
"""

from abc import ABC, abstractmethod
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.path import Path

from openretailscience.options import PlotStyleHelper
from openretailscience.plots.styles.colors import get_named_color
from openretailscience.plots.styles.font_utils import get_font_properties


class BaseRoundedBox(mpatches.PathPatch):
    """Base patch with independent corner rounding for top and bottom."""

    def __init__(
        self,
        xy: tuple[float, float],
        width: float,
        height: float,
        top_radius: float = 0.3,
        bottom_radius: float = 0.3,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Create a box whose top and bottom corners use independent radii.

        Args:
            xy: Bottom-left corner (x, y), not the center.
            width: Box width.
            height: Box height.
            top_radius: Radius of the top corners (default 0.3).
            bottom_radius: Radius of the bottom corners (default 0.3).
            **kwargs: Forwarded to PathPatch.

        """
        x, y = xy

        # Define the path with different corner radii
        verts = []
        codes = []

        # Define corner configurations: (center_x, center_y, radius, start_angle, end_angle, fallback_point, is_first)
        corners = [
            # Bottom left corner
            (x + bottom_radius, y + bottom_radius, bottom_radius, np.pi, 3 * np.pi / 2, (x, y), True),
            # Bottom right corner
            (
                x + width - bottom_radius,
                y + bottom_radius,
                bottom_radius,
                3 * np.pi / 2,
                2 * np.pi,
                (x + width, y),
                False,
            ),
            # Top right corner
            (x + width - top_radius, y + height - top_radius, top_radius, 0, np.pi / 2, (x + width, y + height), False),
            # Top left corner
            (x + top_radius, y + height - top_radius, top_radius, np.pi / 2, np.pi, (x, y + height), False),
        ]

        # Generate all corners
        for center_x, center_y, radius, start_angle, end_angle, fallback_point, is_first in corners:
            corner_verts, corner_codes = self._generate_corner(
                center_x=center_x,
                center_y=center_y,
                radius=radius,
                start_angle=start_angle,
                end_angle=end_angle,
                fallback_point=fallback_point,
                is_first=is_first,
            )
            verts.extend(corner_verts)
            codes.extend(corner_codes)

        codes.append(Path.CLOSEPOLY)
        verts.append((0, 0))

        path = Path(verts, codes)
        super().__init__(path, **kwargs)

    @staticmethod
    def _generate_corner(
        center_x: float,
        center_y: float,
        radius: float,
        start_angle: float,
        end_angle: float,
        fallback_point: tuple[float, float],
        is_first: bool,
    ) -> tuple[list[tuple[float, float]], list[int]]:
        """Emit the vertices and path codes for one corner.

        radius > 0 yields an arc; radius == 0 degenerates to the single fallback_point
        (a straight corner). The first corner uses MOVETO, the rest use LINETO.
        """
        arc_points = 10  # Number of points to use for the corner arc

        if radius > 0:
            theta = np.linspace(start_angle, end_angle, arc_points)

            verts = [(center_x + radius * np.cos(t), center_y + radius * np.sin(t)) for t in theta]
            codes = [Path.MOVETO] + [Path.LINETO] * (arc_points - 1) if is_first else [Path.LINETO] * arc_points

            return verts, codes

        verts = [fallback_point]
        codes = [Path.MOVETO if is_first else Path.LINETO]

        return verts, codes


class TreeNode(ABC):
    """Abstract base class for tree nodes.

    Subclasses must define the NODE_WIDTH and NODE_HEIGHT class attributes and
    implement render(ax).
    """

    # Subclasses must define these class attributes
    NODE_WIDTH: float
    NODE_HEIGHT: float

    def __init__(
        self,
        data: dict[str, Any],
        x: float,
        y: float,
    ) -> None:
        """Create a tree node at a grid position.

        Args:
            data: Node data dict; required keys are defined per subclass.
            x: X-coordinate of the bottom-left corner.
            y: Y-coordinate of the bottom-left corner.

        """
        self._data = data
        self.x = x
        self.y = y

    @abstractmethod
    def render(self, ax: Axes) -> None:
        """Render the node on the given axes.

        Args:
            ax: Matplotlib axes object to render on.

        """
        ...

    def get_width(self) -> float:
        """Return the node width."""
        return self.NODE_WIDTH

    def get_height(self) -> float:
        """Return the node height."""
        return self.NODE_HEIGHT


class SimpleTreeNode(TreeNode):
    """Simple tree node implementation with header and data sections.

    Required data keys:
        header (str): The header text
        percent (float): Percentage change value
        value1 (str): First value text
        value2 (str): Second value text
    """

    NODE_WIDTH = 3.75
    NODE_HEIGHT = 1.75

    # Color thresholds for percent change
    GREEN_THRESHOLD = 1.0  # Percent change at or above this shows green
    RED_THRESHOLD = -1.0  # Percent change at or below this shows red

    @staticmethod
    def _get_color(percent_change: float) -> str:
        """Return the named color for a percent change.

        "positive" (green) at or above GREEN_THRESHOLD (+1.0), "negative" (red) at or
        below RED_THRESHOLD (-1.0), "neutral" otherwise.
        """
        if percent_change >= SimpleTreeNode.GREEN_THRESHOLD:
            return get_named_color("positive")
        if percent_change <= SimpleTreeNode.RED_THRESHOLD:
            return get_named_color("negative")
        return get_named_color("neutral")

    def render(self, ax: Axes) -> None:
        """Render the node on the given axes.

        Args:
            ax: Matplotlib axes object to render on.

        """
        # Extract data from the data dict
        header = self._data["header"]
        percent = self._data["percent"]
        value1 = self._data["value1"]
        value2 = self._data["value2"]

        # Styling constants
        corner_radius = 0.15
        header_height_ratio = 0.4
        header_color = get_named_color("primary")  # Default primary color
        text_color = "white"
        value_vertical_offset = 0.175

        # Positioning fractions
        header_text_x_fraction = 1 / 2
        percent_text_x_fraction = 4 / 16
        value_text_x_fraction = 11 / 16

        style = PlotStyleHelper()

        # Get standard font properties
        semi_bold_font = get_font_properties(style.title_font)
        regular_font = get_font_properties(style.label_font)

        # Determine color based on percent change
        color = self._get_color(percent)

        # Header section
        header_height = self.NODE_HEIGHT * header_height_ratio
        header_box = BaseRoundedBox(
            (self.x, self.y + self.NODE_HEIGHT - header_height),
            self.NODE_WIDTH,
            header_height,
            top_radius=corner_radius,
            bottom_radius=0,
            facecolor=header_color,
            edgecolor="none",
            linewidth=0,
        )
        ax.add_patch(header_box)

        # Data section
        data_height = self.NODE_HEIGHT - header_height
        data_box = BaseRoundedBox(
            (self.x, self.y),
            self.NODE_WIDTH,
            data_height,
            top_radius=0,
            bottom_radius=corner_radius,
            facecolor=color,
            edgecolor="none",
            linewidth=0,
        )
        ax.add_patch(data_box)

        ax.text(
            self.x + self.NODE_WIDTH * header_text_x_fraction,
            self.y + self.NODE_HEIGHT - header_height / 2,
            header,
            ha="center",
            va="center",
            fontproperties=semi_bold_font,
            fontsize=style.label_size,
            color=text_color,
        )

        ax.text(
            self.x + self.NODE_WIDTH * percent_text_x_fraction,
            self.y + data_height / 2,
            f"{percent:+.1f}%",
            ha="center",
            va="center",
            fontproperties=semi_bold_font,
            fontsize=style.title_size,
            color=text_color,
        )

        ax.text(
            self.x + self.NODE_WIDTH * value_text_x_fraction,
            self.y + data_height / 2 + value_vertical_offset,
            value1,
            ha="left",
            va="center",
            fontproperties=regular_font,
            fontsize=style.label_size,
            color=text_color,
        )
        ax.text(
            self.x + self.NODE_WIDTH * value_text_x_fraction,
            self.y + data_height / 2 - value_vertical_offset,
            value2,
            ha="left",
            va="center",
            fontproperties=regular_font,
            fontsize=style.label_size,
            color=text_color,
        )


class TreeGrid:
    """Grid-based tree diagram renderer with configurable node types."""

    # Connection styling constants
    CONNECTION_CURVE_RADIUS = 0.15
    CONNECTION_LINE_WIDTH = 2
    CONNECTION_LINE_COLOR = "black"

    def __init__(
        self,
        tree_structure: dict[str, dict],
        num_rows: int,
        num_cols: int,
        node_class: type[TreeNode],
        vertical_spacing: float | None = None,
        horizontal_spacing: float | None = None,
    ) -> None:
        """Create a grid of nodes with validated positions and a computed layout.

        Each node's ``position`` is ``(col, row)`` where row 0 is the TOP row (rows
        increase downward, the reverse of matplotlib's default y-axis).

        Args:
            tree_structure: Node IDs mapped to node data; each entry needs a
                ``position`` key plus the keys its node_class requires.
            num_rows: Number of grid rows (must be positive).
            num_cols: Number of grid columns (must be positive).
            node_class: TreeNode subclass used to render nodes.
            vertical_spacing: Gap between rows; if None, node_height + 0.6.
            horizontal_spacing: Gap between columns; if None, node_width - 1.0
                (columns overlap for a compact layout).

        Raises:
            ValueError: If grid dimensions are non-positive, tree_structure is empty,
                a node is missing its ``position`` key, or a position is out of bounds.
            TypeError: If node_class is not a TreeNode subclass.

        """
        # Validate grid dimensions
        if num_rows <= 0 or num_cols <= 0:
            error_msg = f"Grid dimensions must be positive: num_rows={num_rows}, num_cols={num_cols}"
            raise ValueError(error_msg)

        # Validate node_class is a TreeNode subclass
        if not issubclass(node_class, TreeNode):
            error_msg = f"node_class must be a TreeNode subclass, got {node_class}"
            raise TypeError(error_msg)

        # Validate tree_structure is not empty
        if not tree_structure:
            raise ValueError("tree_structure cannot be empty")

        self.tree_structure = tree_structure
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.node_class = node_class

        # Get node dimensions from the node class
        self.node_width = node_class.NODE_WIDTH
        self.node_height = node_class.NODE_HEIGHT

        # Auto-calculate spacing if not provided
        self.vertical_spacing = vertical_spacing if vertical_spacing is not None else self.node_height + 0.6
        self.horizontal_spacing = horizontal_spacing if horizontal_spacing is not None else self.node_width - 1.0

        # Validate positions are within grid bounds
        for node_id, node_data in tree_structure.items():
            if "position" not in node_data:
                error_msg = f"Node '{node_id}' is missing required 'position' key"
                raise ValueError(error_msg)

            col_idx, row_idx = node_data["position"]
            if not (0 <= col_idx < num_cols):
                error_msg = f"Node '{node_id}' column index {col_idx} is out of bounds [0, {num_cols})"
                raise ValueError(error_msg)
            if not (0 <= row_idx < num_rows):
                error_msg = f"Node '{node_id}' row index {row_idx} is out of bounds [0, {num_rows})"
                raise ValueError(error_msg)

        # Generate row and column positions
        # Row 0 is at the top, increasing downward (reversed from matplotlib's default bottom-up)
        self.row = {i: (num_rows - 1 - i) * self.vertical_spacing for i in range(num_rows)}
        self.col = {i: i * self.horizontal_spacing for i in range(num_cols)}

    def render(self, ax: Axes | None = None) -> Axes:
        """Render every node and the elbow connector from each parent to its children.

        Args:
            ax: Axes to draw on; if None, a new figure sized to the layout is created.

        Returns:
            Axes: The rendered matplotlib axes.

        Raises:
            ValueError: If a node's ``children`` references an id absent from
                tree_structure.

        """
        if ax is None:
            # Calculate plot dimensions based on layout
            plot_width = self.col[self.num_cols - 1] + self.node_width
            # Row 0 is at the top with the highest y-value after coordinate inversion
            plot_height = self.row[0] + self.node_height

            _, ax = plt.subplots(figsize=(plot_width, plot_height))
            ax.set_xlim(0, plot_width)
            ax.set_ylim(0, plot_height)
            ax.axis("off")

        # First pass: create all nodes and store their centers
        node_centers = {}
        for node_id, node_data in self.tree_structure.items():
            # Convert grid coordinates to absolute positions
            col_idx, row_idx = node_data["position"]
            x = self.col[col_idx]
            y = self.row[row_idx]

            # Extract data for the node (exclude position and children which are structural)
            data_dict = {k: v for k, v in node_data.items() if k not in ("position", "children")}

            # Create and render the node
            node = self.node_class(
                data=data_dict,
                x=x,
                y=y,
            )
            node.render(ax)

            # Store center positions for connections
            node_centers[node_id] = {
                "x": x + self.node_width / 2,
                "y_bottom": y,
                "y_top": y + self.node_height,
            }

        # Second pass: draw connections
        for node_id, node_data in self.tree_structure.items():
            if "children" in node_data and len(node_data["children"]) > 0:
                parent = node_centers[node_id]
                for child_id in node_data["children"]:
                    if child_id not in node_centers:
                        available_nodes = list(node_centers.keys())
                        error_msg = (
                            f"Child node '{child_id}' referenced by '{node_id}' not found in tree_structure. "
                            f"Available nodes: {available_nodes}"
                        )
                        raise ValueError(error_msg)
                    child = node_centers[child_id]
                    self._draw_connection(
                        ax=ax,
                        x1=parent["x"],
                        y1=parent["y_bottom"],
                        x2=child["x"],
                        y2=child["y_top"],
                    )

        return ax

    @staticmethod
    def _add_curve(
        verts: list[tuple[float, float]],
        codes: list[int],
        x: float,
        y: float,
        x_offset: float,
        y_offset: float,
    ) -> None:
        """Append two CURVE3 control points, curving from (x, y) toward its offset end."""
        verts.append((x, y))
        codes.append(Path.CURVE3)
        verts.append((x + x_offset, y + y_offset))
        codes.append(Path.CURVE3)

    @staticmethod
    def _draw_connection(ax: Axes, x1: float, y1: float, x2: float, y2: float) -> None:
        """Draw the parent-bottom-to-child-top connector as an elbow with curved corners."""
        # Use class constants for connection styling
        curve_radius = TreeGrid.CONNECTION_CURVE_RADIUS
        line_width = TreeGrid.CONNECTION_LINE_WIDTH
        line_color = TreeGrid.CONNECTION_LINE_COLOR

        mid_y = (y1 + y2) / 2
        curve_sign = 1 if x2 > x1 else -1

        # Create path with curved corners using Bezier curves
        verts = []
        codes = []

        # Start point (bottom of parent node)
        verts.append((x1, y1))
        codes.append(Path.MOVETO)

        # Vertical line down to curve start
        verts.append((x1, mid_y + curve_radius))
        codes.append(Path.LINETO)

        # Curve from vertical to horizontal (first corner)
        TreeGrid._add_curve(verts, codes, x1, mid_y, curve_sign * curve_radius, 0)

        # Horizontal line
        verts.append((x2 - (curve_sign * curve_radius), mid_y))
        codes.append(Path.LINETO)

        # Curve from horizontal to vertical (second corner)
        TreeGrid._add_curve(verts, codes, x2, mid_y, 0, -curve_radius)

        # Vertical line up to child node
        verts.append((x2, y2))
        codes.append(Path.LINETO)

        # Create and draw the path
        path = Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor="none", edgecolor=line_color, linewidth=line_width)
        ax.add_patch(patch)


class DetailedTreeNode(TreeNode):
    """Detailed tree node showing current/previous period, diff, pct diff, and contribution.

    Required data keys:
        header: str - Node header text
        percent: float - Percentage change; also drives the "Pct Diff" row and header color
        current_period: str - Current period value text
        previous_period: str - Previous period value text
        diff: str - Absolute difference text

    Optional data keys:
        contribution: str - Contribution value (row left blank if absent)
        current_label: str - Current period label (default "Current Period")
        previous_label: str - Previous period label (default "Previous Period")
    """

    NODE_WIDTH = 3.5
    NODE_HEIGHT = 2.5

    # Color thresholds for percent change
    GREEN_THRESHOLD = 1.0  # Percent change at or above this shows green
    RED_THRESHOLD = -1.0  # Percent change at or below this shows red

    @staticmethod
    def _get_color(percent_change: float) -> str:
        """Return the named color for a percent change.

        "positive" (green) at or above GREEN_THRESHOLD (+1.0), "negative" (red) at or
        below RED_THRESHOLD (-1.0), "neutral" otherwise.
        """
        if percent_change >= DetailedTreeNode.GREEN_THRESHOLD:
            return get_named_color("positive")
        if percent_change <= DetailedTreeNode.RED_THRESHOLD:
            return get_named_color("negative")
        return get_named_color("neutral")

    def render(self, ax: Axes) -> None:
        """Render the detailed node on the given axes.

        Args:
            ax: Matplotlib axes object to render on.

        """
        # Extract data from the data dict
        header = self._data["header"]
        percent = self._data["percent"]
        current_period = self._data["current_period"]
        previous_period = self._data["previous_period"]
        diff = self._data["diff"]
        contribution = self._data.get("contribution")
        # Styling constants
        corner_radius = 0.15
        header_height_ratio = 0.25
        header_color = self._get_color(percent)
        header_text_color = "white"
        data_bg_color = "white"
        data_text_color = "#4b5563"
        label_color = "#6b7280"
        border_color = "#e5e7eb"

        style = PlotStyleHelper()

        # Get standard font properties
        semi_bold_font = get_font_properties(style.title_font)
        regular_font = get_font_properties(style.label_font)

        header_font_size = 12
        period_metric_font_size = 12
        info_metric_font_size = 10
        small_label_font_size = 10

        # Title section (colored header)
        title_section_height = self.NODE_HEIGHT * header_height_ratio
        data_section_height = self.NODE_HEIGHT - title_section_height

        # Layout configuration based on visual structure (all values are fixed/absolute)
        layout = {
            "node": {
                "padding_left": 0.2,
                "padding_right": 0.2,
            },
            "data_section": {
                "padding_top": 0.2,  # Space above period labels
                "padding_bottom": 0.1,  # Space below last info row
                "period_subsection": {
                    "previous_period_x": (self.NODE_WIDTH / 2) + 0.2,  # X position for "Previous Period" column
                    "label_row_gap": 0.25,  # Vertical gap from labels to metrics
                    "bottom_margin": 0.25,  # Space below period metrics before divider
                },
                "divider": {
                    "horizontal_inset": 0.2,  # Padding on left/right of divider line
                    "vertical_offset": 0.025,  # Fine-tune vertical position
                },
                "info_subsection": {
                    "row_count": 3,  # Number of info rows
                    "row_spacing_scale": 0.75,  # Scale factor for vertical spacing between rows
                },
            },
        }
        # Render title section box (border matches header color for seamless appearance)
        title_box = BaseRoundedBox(
            xy=(self.x, self.y + self.NODE_HEIGHT - title_section_height),
            width=self.NODE_WIDTH,
            height=title_section_height,
            top_radius=corner_radius,
            bottom_radius=0,
            facecolor=header_color,
            edgecolor=header_color,
            linewidth=1,
        )
        ax.add_patch(title_box)

        # Render data section box
        data_section_box = BaseRoundedBox(
            xy=(self.x, self.y),
            width=self.NODE_WIDTH,
            height=data_section_height,
            top_radius=0,
            bottom_radius=corner_radius,
            facecolor=data_bg_color,
            edgecolor=border_color,
            linewidth=1,
        )
        ax.add_patch(data_section_box)

        # Calculate horizontal positions
        content_left_x = self.x + layout["node"]["padding_left"]
        previous_period_x = self.x + layout["data_section"]["period_subsection"]["previous_period_x"]
        content_right_x = self.x + self.NODE_WIDTH - layout["node"]["padding_right"]

        # Render title text
        ax.text(
            content_left_x,
            self.y + self.NODE_HEIGHT - title_section_height / 2,
            header,
            ha="left",
            va="center",
            fontproperties=semi_bold_font,
            fontsize=header_font_size,
            color=header_text_color,
        )

        # Calculate vertical positions for data section
        data_section_top = self.y + data_section_height

        # Draw horizontal line separator between title and data sections
        title_data_separator_y = self.y + data_section_height
        ax.plot(
            [self.x, self.x + self.NODE_WIDTH],
            [title_data_separator_y, title_data_separator_y],
            color=border_color,
            linewidth=1.5,
            zorder=10,
        )

        # Period subsection: labels and metrics
        period_labels_y = data_section_top - layout["data_section"]["padding_top"]
        period_metrics_y = period_labels_y - layout["data_section"]["period_subsection"]["label_row_gap"]

        # Divider line between period and info subsections
        divider_y = (
            period_metrics_y
            - layout["data_section"]["period_subsection"]["bottom_margin"]
            + layout["data_section"]["divider"]["vertical_offset"]
        )
        divider_inset = layout["data_section"]["divider"]["horizontal_inset"]
        ax.plot(
            [self.x + divider_inset, self.x + self.NODE_WIDTH - divider_inset],
            [divider_y, divider_y],
            color=border_color,
            linewidth=1,
            zorder=10,
        )

        # Get period labels (with defaults)
        current_label = self._data.get("current_label", "Current Period")
        previous_label = self._data.get("previous_label", "Previous Period")

        # Render period subsection (labels and metrics for current and previous periods)
        period_columns = [
            (current_label, current_period, content_left_x),
            (previous_label, previous_period, previous_period_x),
        ]

        for label, metric, x_pos in period_columns:
            # Period label
            ax.text(
                x_pos,
                period_labels_y,
                label,
                ha="left",
                va="center",
                fontproperties=regular_font,
                fontsize=small_label_font_size,
                color=label_color,
            )
            # Period metric
            ax.text(
                x_pos,
                period_metrics_y,
                metric,
                ha="left",
                va="center",
                fontproperties=semi_bold_font,
                fontsize=period_metric_font_size,
                color=data_text_color,
            )

        # Calculate remaining height for info subsection
        info_section_height = (
            data_section_height
            - layout["data_section"]["padding_top"]
            - layout["data_section"]["period_subsection"]["label_row_gap"]
            - layout["data_section"]["period_subsection"]["bottom_margin"]
            - layout["data_section"]["padding_bottom"]
        )
        info_row_height = info_section_height / layout["data_section"]["info_subsection"]["row_count"]

        # Render info subsection rows (Diff, Pct Diff, Contribution)
        info_rows = [
            ("Diff", diff),
            ("Pct Diff", f"{percent:.1f}%"),
        ]
        # Only add contribution row if it's provided
        if contribution is not None:
            info_rows.append(("Contribution", contribution))

        for i, (label, metric) in enumerate(info_rows):
            info_row_y = divider_y - info_row_height * (
                i + layout["data_section"]["info_subsection"]["row_spacing_scale"]
            )
            # Label on left
            ax.text(
                content_left_x,
                info_row_y,
                label,
                ha="left",
                va="center",
                fontproperties=regular_font,
                fontsize=small_label_font_size,
                color=label_color,
            )
            # Metric on right
            ax.text(
                content_right_x,
                info_row_y,
                metric,
                ha="right",
                va="center",
                fontproperties=semi_bold_font,
                fontsize=info_metric_font_size,
                color=data_text_color,
            )
