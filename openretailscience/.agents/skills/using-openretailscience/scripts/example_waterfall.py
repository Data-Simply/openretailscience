"""Example script demonstrating waterfall plot usage."""

from openretailscience.plots import waterfall

# Example 1: Revenue change breakdown
labels = ["Starting Revenue", "New Customers", "Upsells", "Price Increase", "Churn", "Returns"]
amounts = [1000000, 250000, 150000, 80000, -180000, -50000]

waterfall.plot(
    amounts=amounts,
    labels=labels,
    title="Revenue Change Breakdown FY2024 vs FY2023",
    y_label="Revenue ($)",
    data_label_format="absolute",
    display_net_bar=True,
    source_text="Source: Annual Financial Report",
    rot=15,
)

# Example 2: With percentage labels
labels2 = ["Gross Profit", "Marketing", "Operations", "R&D", "Admin"]
amounts2 = [2000000, -450000, -600000, -300000, -150000]

waterfall.plot(
    amounts=amounts2,
    labels=labels2,
    title="Profit Margin Breakdown",
    y_label="Amount ($)",
    data_label_format="percentage",
    display_net_bar=True,
    display_net_line=True,
    source_text="Source: Q4 2024 Financial Results",
)

# Example 3: Both absolute and percentage
labels3 = ["Base Sales", "Promo Impact", "New Products", "Seasonal Decline", "Competition"]
amounts3 = [500000, 125000, 75000, -60000, -40000]

waterfall.plot(
    amounts=amounts3,
    labels=labels3,
    title="Sales Drivers Analysis",
    y_label="Sales ($)",
    data_label_format="both",
    display_net_bar=True,
    rot=0,
)

# Example 6: Simple without net bar
labels6 = ["Category A", "Category B", "Category C", "Category D"]
amounts6 = [150000, -45000, 78000, 32000]

waterfall.plot(
    amounts=amounts6,
    labels=labels6,
    title="Category Contributions",
    y_label="Impact ($)",
    data_label_format="percentage",
    display_net_bar=False,
    display_net_line=True,
)
