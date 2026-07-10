"""Example script demonstrating tree diagram usage.

This script shows how to create hierarchical tree diagrams for
period-over-period comparisons using the matplotlib-based TreeGrid
renderer from openretailscience.
"""

from openretailscience.plots.tree_diagram import DetailedTreeNode, SimpleTreeNode, TreeGrid

# Example 1: Simple revenue decomposition (DetailedTreeNode)
tree_structure_1 = {
    "total_revenue": {
        "position": (1, 0),
        "children": ["customers", "revenue_per_customer"],
        "header": "Total Revenue",
        "percent": 25.0,
        "current_period": "$1.5M",
        "previous_period": "$1.2M",
        "diff": "$300K",
        "current_label": "FY2024",
        "previous_label": "FY2023",
    },
    "customers": {
        "position": (0, 1),
        "header": "Number of Customers",
        "percent": 11.1,
        "current_period": "5,000",
        "previous_period": "4,500",
        "diff": "500",
    },
    "revenue_per_customer": {
        "position": (2, 1),
        "header": "Revenue per Customer",
        "percent": 12.4,
        "current_period": "$300",
        "previous_period": "$267",
        "diff": "$33",
    },
}

grid_1 = TreeGrid(
    tree_structure=tree_structure_1,
    num_rows=2,
    num_cols=3,
    node_class=DetailedTreeNode,
)
grid_1.render()

# Example 2: Same tree, with an optional "contribution" key added to each node
tree_structure_2 = {
    "total_revenue": {**tree_structure_1["total_revenue"], "contribution": "100%"},
    "customers": {**tree_structure_1["customers"], "contribution": "62%"},
    "revenue_per_customer": {**tree_structure_1["revenue_per_customer"], "contribution": "38%"},
}

grid_2 = TreeGrid(
    tree_structure=tree_structure_2,
    num_rows=2,
    num_cols=3,
    node_class=DetailedTreeNode,
)
grid_2.render()

# Example 3: Same tree, with "customers" given its own children for a 3-level hierarchy
tree_structure_3 = {
    "total_revenue": tree_structure_1["total_revenue"],
    "revenue_per_customer": tree_structure_1["revenue_per_customer"],
    "customers": {**tree_structure_1["customers"], "children": ["returning_customers", "new_customers"]},
    "returning_customers": {
        "position": (0, 2),
        "header": "Returning Customers",
        "percent": 8.3,
        "current_period": "3,900",
        "previous_period": "3,600",
        "diff": "300",
    },
    "new_customers": {
        "position": (1, 2),
        "header": "New Customers",
        "percent": 22.2,
        "current_period": "1,100",
        "previous_period": "900",
        "diff": "200",
    },
}

grid_3 = TreeGrid(
    tree_structure=tree_structure_3,
    num_rows=3,
    num_cols=3,
    node_class=DetailedTreeNode,
)
grid_3.render()

# Example 4: Simple node layout (SimpleTreeNode)
tree_structure_4 = {
    "margin": {
        "position": (1, 0),
        "children": ["revenue_growth"],
        "header": "Profit Margin %",
        "percent": 27.0,
        "value1": "15.68%",
        "value2": "12.35%",
    },
    "revenue_growth": {
        "position": (1, 1),
        "header": "Revenue Growth",
        "percent": 25.0,
        "value1": "$1.23M",
        "value2": "$0.99M",
    },
}

grid_4 = TreeGrid(
    tree_structure=tree_structure_4,
    num_rows=2,
    num_cols=3,
    node_class=SimpleTreeNode,
)
grid_4.render()
