# ============================================================
# CUSTOMER REVENUE, RETENTION & SEGMENTATION ANALYTICS
# ============================================================

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ucimlrepo import fetch_ucirepo


# ============================================================
# 1. CONFIGURATION
# ============================================================

print("=" * 70)
print("CUSTOMER REVENUE, RETENTION & SEGMENTATION ANALYTICS")
print("=" * 70)


RANDOM_SEED = 42


ROOT = Path(".")

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

FIGURE_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"


for directory in [
    RAW_DIR,
    PROCESSED_DIR,
    FIGURE_DIR,
    TABLE_DIR
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 2. DOWNLOAD UCI ONLINE RETAIL DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 — DOWNLOADING UCI ONLINE RETAIL DATA")
print("=" * 70)


online_retail = fetch_ucirepo(
    id=352
)


raw = (
    online_retail
    .data
    .original
    .copy()
)

print("\nRaw observations:")
print(len(raw))
print("\nRaw columns:")
print(
    list(
        raw.columns
    )
)
# ============================================================
# 3. STANDARDISE DATA TYPES AND COLUMN NAMES
# ============================================================
rename_map = {

    "InvoiceNo": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "UnitPrice": "unit_price",
    "CustomerID": "customer_id",
    "Country": "country"
}
raw = raw.rename(
    columns=rename_map
)
required_columns = {
    "invoice_no",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "unit_price",
    "customer_id",
    "country"
}
missing_columns = (
    required_columns
    - set(
        raw.columns
    )
)
if missing_columns:

    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )
raw["invoice_date"] = pd.to_datetime(
    raw["invoice_date"],
    errors="coerce"
)
raw["invoice_no"] = (
    raw["invoice_no"]
    .astype(str)
)
raw["stock_code"] = (
    raw["stock_code"]
    .astype(str)
)
raw["quantity"] = pd.to_numeric(
    raw["quantity"],
    errors="coerce"
)
raw["unit_price"] = pd.to_numeric(
    raw["unit_price"],
    errors="coerce"
)
raw["line_revenue"] = (
    raw["quantity"]
    * raw["unit_price"]
)
raw["is_cancelled"] = (
    raw["invoice_no"]
    .str
    .upper()
    .str
    .startswith("C")
)
raw.to_csv(
    RAW_DIR / "online_retail_raw.csv",
    index=False
)
# ============================================================
# 4. DATA QUALITY AUDIT
# ============================================================
print("\n" + "=" * 70)
print("STEP 2 — DATA QUALITY AUDIT")
print("=" * 70)
print("\nMissing values:")
print(
    raw
    .isna()
    .sum()
)
total_invoices = (
    raw["invoice_no"]
    .nunique()
)
cancelled_invoices = (
    raw.loc[
        raw["is_cancelled"],
        "invoice_no"
    ]
    .nunique()
)
cancellation_rate = (
    cancelled_invoices
    / total_invoices
    if total_invoices > 0
    else np.nan
)
print("\nTotal invoices:")
print(total_invoices)
print("\nCancelled invoices:")
print(cancelled_invoices)
print("\nInvoice cancellation rate:")
print(
    round(
        cancellation_rate * 100,
        2
    ),
    "%"
)
duplicate_rows = int(
    raw.duplicated().sum()
)
print("\nExact duplicate transaction rows:")
print(duplicate_rows)
raw = (
    raw
    .drop_duplicates()
    .reset_index(drop=True)
)
# ============================================================
# 5. CLEAN VALID SALES TRANSACTIONS
# ============================================================
print("\n" + "=" * 70)
print("STEP 3 — CLEANING VALID SALES")
print("=" * 70)
transactions = raw.copy()
transactions = transactions[
    (~transactions["is_cancelled"])
    & (
        transactions["quantity"] > 0
    )
    & (
        transactions["unit_price"] > 0
    )
    & (
        transactions["customer_id"].notna()
    )
    & (
        transactions["invoice_date"].notna()
    )
].copy()
transactions["customer_id"] = (
    pd.to_numeric(
        transactions["customer_id"],
        errors="coerce"
    )
    .astype("Int64")
    .astype(str)
)
transactions["description"] = (
    transactions["description"]
    .fillna(
        "Unknown Product"
    )
    .astype(str)
)
transactions["country"] = (
    transactions["country"]
    .fillna(
        "Unknown"
    )
    .astype(str)
)
transactions["line_revenue"] = (
    transactions["quantity"]
    * transactions["unit_price"]
)
transactions = transactions[
    transactions["line_revenue"] > 0
].copy()
transactions = (
    transactions
    .sort_values(
        "invoice_date"
    )
    .reset_index(
        drop=True
    )
)
transactions.to_csv(
    PROCESSED_DIR / "clean_transactions.csv",
    index=False
)
print("\nValid sales transaction rows:")
print(len(transactions))
print("\nUnique customers:")
print(
    transactions[
        "customer_id"
    ]
    .nunique()
)
print("\nUnique valid orders:")
print(
    transactions[
        "invoice_no"
    ]
    .nunique()
)
print("\nTransaction date range:")
print(
    transactions[
        "invoice_date"
    ]
    .min(),
    "→",
    transactions[
        "invoice_date"
    ]
    .max()
)
# ============================================================
# 6. CREATE ORDER-LEVEL DATASET
# ============================================================
orders = (
    transactions
    .groupby(
        [
            "invoice_no",
            "customer_id"
        ],
        as_index=False
    )
    .agg(
        invoice_date=(
            "invoice_date",
            "min"
        ),
        country=(
            "country",
            "first"
        ),
        order_revenue=(
            "line_revenue",
            "sum"
        ),
        units=(
            "quantity",
            "sum"
        ),
        product_lines=(
            "stock_code",
            "count"
        ),
        unique_products=(
            "stock_code",
            "nunique"
        )
    )
)
orders.to_csv(
    PROCESSED_DIR / "orders.csv",
    index=False
)
# ============================================================
# 7. DUCKDB + SQL ANALYTICS
# ============================================================
print("\n" + "=" * 70)
print("STEP 4 — DUCKDB + SQL BUSINESS ANALYTICS")
print("=" * 70)
con = duckdb.connect(
    "customer_analytics.duckdb"
)
con.register(
    "transactions_df",
    transactions
)
con.register(
    "orders_df",
    orders
)
con.execute(
    """
    CREATE OR REPLACE TABLE transactions
    AS
    SELECT *
    FROM transactions_df
    """
)
con.execute(
    """
    CREATE OR REPLACE TABLE orders
    AS
    SELECT *
    FROM orders_df
    """
)
# ============================================================
# 8. CORE BUSINESS KPIs
# ============================================================
kpis = con.execute(
    """
    SELECT
        SUM(order_revenue)
            AS total_revenue,
        COUNT(DISTINCT invoice_no)
            AS orders,
        COUNT(DISTINCT customer_id)
            AS customers,
        SUM(units)
            AS units_sold,
        AVG(order_revenue)
            AS average_order_value
    FROM orders
    """
).df()
total_revenue = float(
    kpis.loc[
        0,
        "total_revenue"
    ]
)
total_orders = int(
    kpis.loc[
        0,
        "orders"
    ]
)
total_customers = int(
    kpis.loc[
        0,
        "customers"
    ]
)
average_order_value = float(
    kpis.loc[
        0,
        "average_order_value"
    ]
)
customer_order_counts = (
    orders
    .groupby(
        "customer_id"
    )[
        "invoice_no"
    ]
    .nunique()
)
repeat_customer_rate = (
    customer_order_counts
    .ge(2)
    .mean()
)
kpis[
    "repeat_customer_rate"
] = repeat_customer_rate
kpis[
    "invoice_cancellation_rate"
] = cancellation_rate
kpis.to_csv(
    TABLE_DIR / "business_kpis.csv",
    index=False
)
print("\nCore business KPIs:")
print(
    kpis
    .round(4)
    .to_string(
        index=False
    )
)
# ============================================================
# 9. MONTHLY BUSINESS PERFORMANCE
# ============================================================
monthly_kpis = con.execute(
    """
    SELECT
        DATE_TRUNC(
            'month',
            invoice_date
        ) AS month,
        SUM(order_revenue)
            AS revenue,
        COUNT(DISTINCT invoice_no)
            AS orders,
        COUNT(DISTINCT customer_id)
            AS active_customers,
        AVG(order_revenue)
            AS average_order_value
    FROM orders
    GROUP BY month
    ORDER BY month
    """
).df()
monthly_kpis.to_csv(
    TABLE_DIR / "monthly_kpis.csv",
    index=False
)
print("\nMonthly KPIs:")
monthly_kpis_display = (
    monthly_kpis.copy()
)
monthly_kpis_display[
    "revenue"
] = (
    monthly_kpis_display[
        "revenue"
    ]
    .round(2)
)
monthly_kpis_display[
    "average_order_value"
] = (
    monthly_kpis_display[
        "average_order_value"
    ]
    .round(2)
)
print(
    monthly_kpis_display
    .to_string(
        index=False
    )
)
# ============================================================
# 10. COUNTRY PERFORMANCE
# ============================================================
country_performance = con.execute(
    """
    SELECT
        country,
        SUM(order_revenue)
            AS revenue,
        COUNT(DISTINCT invoice_no)
            AS orders,
        COUNT(DISTINCT customer_id)
            AS customers
    FROM orders
    GROUP BY country
    ORDER BY revenue DESC
    """
).df()
country_performance[
    "revenue_share"
] = (
    country_performance[
        "revenue"
    ]
    / country_performance[
        "revenue"
    ].sum()
)
country_performance.to_csv(
    TABLE_DIR / "country_performance.csv",
    index=False
)
print("\nTop countries by revenue:")
print(
    country_performance
    .head(10)
    .round(4)
    .to_string(
        index=False
    )
)
# ============================================================
# 11. PRODUCT PERFORMANCE
# ============================================================
NON_MERCHANDISE_CODES = [
    "POST",
    "M",
    "DOT",
    "BANK CHARGES",
    "AMAZONFEE",
    "CRUK",
    "D"
]
merchandise_transactions = (
    transactions[
        ~transactions[
            "stock_code"
        ].isin(
            NON_MERCHANDISE_CODES
        )
    ]
    .copy()
)
con.register(
    "merchandise_transactions_df",
    merchandise_transactions
)
con.execute(
    """
    CREATE OR REPLACE TABLE merchandise_transactions
    AS
    SELECT *
    FROM merchandise_transactions_df
    """
)
product_performance = con.execute(
    """
    SELECT
        stock_code,
        description,
        SUM(quantity)
            AS units_sold,
        SUM(line_revenue)
            AS revenue,
        COUNT(DISTINCT invoice_no)
            AS orders
    FROM merchandise_transactions
    GROUP BY
        stock_code,
        description
    ORDER BY revenue DESC
    """
).df()
product_performance[
    "order_penetration"
] = (
    product_performance[
        "orders"
    ]
    / total_orders
)
product_performance.to_csv(
    TABLE_DIR / "product_performance.csv",
    index=False
)
print("\nTop 10 products by revenue:")
print(
    product_performance
    .head(10)
    .round(4)
    .to_string(
        index=False
    )
)
# ============================================================
# 12. CUSTOMER RFM ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("STEP 5 — CUSTOMER RFM ANALYSIS")
print("=" * 70)
snapshot_date = (
    transactions[
        "invoice_date"
    ]
    .max()
    + pd.Timedelta(
        days=1
    )
)
customer_rfm = (
    orders
    .groupby(
        "customer_id"
    )
    .agg(
        first_purchase=(
            "invoice_date",
            "min"
        ),
        last_purchase=(
            "invoice_date",
            "max"
        ),
        frequency=(
            "invoice_no",
            "nunique"
        ),
        monetary=(
            "order_revenue",
            "sum"
        ),
        average_order_value=(
            "order_revenue",
            "mean"
        )
    )
    .reset_index()
)
customer_rfm[
    "recency_days"
] = (
    snapshot_date
    - customer_rfm[
        "last_purchase"
    ]
).dt.days
customer_rfm[
    "tenure_days"
] = (
    customer_rfm[
        "last_purchase"
    ]
    - customer_rfm[
        "first_purchase"
    ]
).dt.days
# ============================================================
# 13. QUANTILE RFM SCORING
# ============================================================
customer_rfm[
    "r_score"
] = pd.qcut(
    customer_rfm[
        "recency_days"
    ]
    .rank(
        method="first"
    ),
    q=4,
    labels=[
        4,
        3,
        2,
        1
    ]
).astype(int)
customer_rfm[
    "f_score"
] = pd.qcut(

    customer_rfm[
        "frequency"
    ]
    .rank(
        method="first"
    ),

    q=4,

    labels=[
        1,
        2,
        3,
        4
    ]
).astype(int)
customer_rfm[
    "m_score"
] = pd.qcut(
    customer_rfm[
        "monetary"
    ]
    .rank(
        method="first"
    ),
    q=4,
    labels=[
        1,
        2,
        3,
        4
    ]
).astype(int)
customer_rfm[
    "rfm_score"
] = (
    customer_rfm[
        "r_score"
    ]
    + customer_rfm[
        "f_score"
    ]
    + customer_rfm[
        "m_score"
    ]
)
# ============================================================
# 14. BUSINESS RFM SEGMENTS
# ============================================================

def assign_business_segment(row):

    if (
        row["r_score"] >= 4
        and row["f_score"] >= 4
        and row["m_score"] >= 3
    ):
        return "Champions"
    if (
        row["r_score"] >= 3
        and row["f_score"] >= 3
    ):
        return "Loyal Customers"
    if (
        row["r_score"] <= 2
        and row["f_score"] >= 3
    ):
        return "At Risk"
    if (
        row["r_score"] >= 3
        and row["f_score"] <= 2
    ):
        return "New / Promising"
    if row["m_score"] >= 4:
        return "High Value"
    return "Other"
customer_rfm[
    "business_segment"
] = customer_rfm.apply(
    assign_business_segment,
    axis=1
)
# ============================================================
# 15. K-MEANS CUSTOMER SEGMENTATION
# ============================================================

segmentation_features = (
    customer_rfm[
        [
            "recency_days",
            "frequency",
            "monetary"
        ]
    ]
    .copy()
)
segmentation_log = np.log1p(
    segmentation_features
)
scaler = StandardScaler()
segmentation_scaled = (
    scaler.fit_transform(
        segmentation_log
    )
)
silhouette_rows = []
for k in range(
    2,
    7
):
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        n_init=20
    )
    labels = (
        model.fit_predict(
            segmentation_scaled
        )
    )
    score = silhouette_score(
        segmentation_scaled,
        labels
    )
    silhouette_rows.append(
        {
            "clusters": k,
            "silhouette_score": score
        }
    )
silhouette_results = pd.DataFrame(
    silhouette_rows
)
silhouette_results.to_csv(
    TABLE_DIR / "silhouette_scores.csv",
    index=False
)
best_k = int(
    silhouette_results
    .sort_values(
        "silhouette_score",
        ascending=False
    )
    .iloc[0][
        "clusters"
    ]
)
print("\nSilhouette results:")
print(
    silhouette_results
    .round(4)
    .to_string(
        index=False
    )
)
print("\nSelected K-Means cluster count:")
print(best_k)
final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_SEED,
    n_init=20
)
customer_rfm[
    "kmeans_cluster"
] = (
    final_kmeans
    .fit_predict(
        segmentation_scaled
    )
)
customer_rfm.to_csv(
    TABLE_DIR / "customer_rfm_segments.csv",
    index=False
)
cluster_profile = (
    customer_rfm
    .groupby(
        "kmeans_cluster",
        as_index=False
    )
    .agg(
        customers=(
            "customer_id",
            "nunique"
        ),
        average_recency_days=(
            "recency_days",
            "mean"
        ),
        average_frequency=(
            "frequency",
            "mean"
        ),
        average_monetary_value=(
            "monetary",
            "mean"
        ),
        total_revenue=(
            "monetary",
            "sum"
        )
    )
)
cluster_profile[
    "customer_share"
] = (
    cluster_profile[
        "customers"
    ]
    / cluster_profile[
        "customers"
    ].sum()
)
cluster_profile[
    "revenue_share"
] = (
    cluster_profile[
        "total_revenue"
    ]
    / cluster_profile[
        "total_revenue"
    ].sum()
)
cluster_profile.to_csv(
    TABLE_DIR / "customer_cluster_profile.csv",
    index=False
)
print("\nK-Means customer profiles:")
print(
    cluster_profile
    .round(4)
    .to_string(
        index=False
    )
)
# ============================================================
# 16. BUSINESS SEGMENT SUMMARY
# ============================================================
business_segment_summary = (
    customer_rfm
    .groupby(
        "business_segment",
        as_index=False
    )
    .agg(
        customers=(
            "customer_id",
            "nunique"
        ),
        total_revenue=(
            "monetary",
            "sum"
        ),
        average_revenue=(
            "monetary",
            "mean"
        ),
        average_frequency=(
            "frequency",
            "mean"
        ),
        average_recency_days=(
            "recency_days",
            "mean"
        )
    )
)
business_segment_summary[
    "customer_share"
] = (
    business_segment_summary[
        "customers"
    ]
    / business_segment_summary[
        "customers"
    ].sum()
)
business_segment_summary[
    "revenue_share"
] = (
    business_segment_summary[
        "total_revenue"
    ]
    / business_segment_summary[
        "total_revenue"
    ].sum()
)
business_segment_summary.to_csv(
    TABLE_DIR / "business_segment_summary.csv",
    index=False
)
print("\nBusiness customer segments:")
print(
    business_segment_summary
    .sort_values(
        "total_revenue",
        ascending=False
    )
    .round(4)
    .to_string(
        index=False
    )
)
# ============================================================
# 17. COHORT RETENTION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("STEP 6 — COHORT RETENTION ANALYSIS")
print("=" * 70)
cohort_orders = (
    orders[
        [
            "customer_id",
            "invoice_no",
            "invoice_date"
        ]
    ]
    .copy()
)
cohort_orders[
    "transaction_month"
] = (
    cohort_orders[
        "invoice_date"
    ]
    .dt
    .to_period("M")
)
first_purchase_month = (
    cohort_orders
    .groupby(
        "customer_id"
    )[
        "transaction_month"
    ]
    .min()
)
cohort_orders[
    "cohort_month"
] = (
    cohort_orders[
        "customer_id"
    ]
    .map(
        first_purchase_month
    )
)
cohort_orders[
    "cohort_index"
] = (
    (
        cohort_orders[
            "transaction_month"
        ].dt.year
        - cohort_orders[
            "cohort_month"
        ].dt.year
    )
    * 12
    + (
        cohort_orders[
            "transaction_month"
        ].dt.month
        - cohort_orders[
            "cohort_month"
        ].dt.month
    )
    + 1
)
cohort_counts = (
    cohort_orders
    .groupby(
        [
            "cohort_month",
            "cohort_index"
        ]
    )[
        "customer_id"
    ]
    .nunique()
    .unstack()
)
cohort_sizes = (
    cohort_counts[1]
)
retention_matrix = (
    cohort_counts
    .divide(
        cohort_sizes,
        axis=0
    )
)
# Distinguish genuine zero retention from periods
# that have not yet been observed.
last_transaction_month = (
    cohort_orders[
        "transaction_month"
    ]
    .max()
)
for cohort_month in retention_matrix.index:
    max_observable_index = (
        (
            last_transaction_month.year
            - cohort_month.year
        )
        * 12
        + (
            last_transaction_month.month
            - cohort_month.month
        )
        + 1
    )
    observed_columns = [
        column
        for column
        in retention_matrix.columns
        if column
        <= max_observable_index
    ]
    retention_matrix.loc[
        cohort_month,
        observed_columns
    ] = (
        retention_matrix.loc[
            cohort_month,
            observed_columns
        ]
        .fillna(0)
    )
retention_export = (
    retention_matrix.copy()
)
retention_export.index = (
    retention_export
    .index
    .astype(str)
)
retention_export.to_csv(
    TABLE_DIR / "cohort_retention.csv"
)
print("\nCohort retention matrix:")
print(
    retention_matrix
    .round(3)
    .to_string()
)
# ============================================================
# 18. CUSTOMER REVENUE CONCENTRATION
# ============================================================
customer_revenue = (
    customer_rfm[
        [
            "customer_id",
            "monetary"
        ]
    ]
    .sort_values(
        "monetary",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)
customer_revenue[
    "customer_rank"
] = (
    np.arange(
        1,
        len(
            customer_revenue
        ) + 1
    )
)
customer_revenue[
    "customer_share"
] = (
    customer_revenue[
        "customer_rank"
    ]
    / len(
        customer_revenue
    )
)
customer_revenue[
    "cumulative_revenue"
] = (
    customer_revenue[
        "monetary"
    ]
    .cumsum()
)
customer_revenue[
    "cumulative_revenue_share"
] = (
    customer_revenue[
        "cumulative_revenue"
    ]
    / customer_revenue[
        "monetary"
    ].sum()
)
top_10_percent_count = max(
    1,
    int(
        np.ceil(
            len(
                customer_revenue
            )
            * 0.10
        )
    )
)
top_10_customer_revenue_share = (
    customer_revenue
    .head(
        top_10_percent_count
    )[
        "monetary"
    ]
    .sum()
    / customer_revenue[
        "monetary"
    ].sum()
)
customer_revenue.to_csv(
    TABLE_DIR / "customer_revenue_concentration.csv",
    index=False
)
print(
    "\nRevenue share generated by top 10% of customers:"
)
print(
    round(
        top_10_customer_revenue_share * 100,
        2
    ),
    "%"
)
# ============================================================
# 19. MONTHLY REVENUE FIGURE
# ============================================================
fig, ax = plt.subplots(
    figsize=(12, 6)
)
ax.plot(
    pd.to_datetime(
        monthly_kpis[
            "month"
        ]
    ),
    monthly_kpis[
        "revenue"
    ],
    marker="o"
)
ax.set_title(
    "Monthly Revenue"
)
ax.set_xlabel(
    "Month"
)
ax.set_ylabel(
    "Revenue (£)"
)
ax.grid(
    alpha=0.25
)
fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "monthly_revenue.png",
    dpi=200
)
plt.close(
    fig
)
# ============================================================
# 20. COHORT RETENTION HEATMAP
# ============================================================
retention_plot = (

    retention_matrix

    .iloc[
        :,
        :12
    ]
)
fig, ax = plt.subplots(
    figsize=(12, 8)
)
image = ax.imshow(
    retention_plot.values,
    aspect="auto",
    vmin=0,
    vmax=1
)
ax.set_title(
    "Monthly Customer Cohort Retention"
)
ax.set_xlabel(
    "Months Since First Purchase"
)
ax.set_ylabel(
    "Acquisition Cohort"
)
ax.set_xticks(
    range(
        len(
            retention_plot.columns
        )
    )
)
ax.set_xticklabels(
    retention_plot.columns
)
ax.set_yticks(
    range(
        len(
            retention_plot.index
        )
    )
)
ax.set_yticklabels(
    retention_plot
    .index
    .astype(str)
)
fig.colorbar(
    image,
    ax=ax,
    label="Retention Rate"
)
fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "cohort_retention_heatmap.png",
    dpi=200
)
plt.close(
    fig
)
# ============================================================
# 21. CUSTOMER SEGMENT REVENUE FIGURE
# ============================================================
segment_chart = (

    business_segment_summary

    .sort_values(
        "total_revenue",
        ascending=True
    )
)
fig, ax = plt.subplots(
    figsize=(10, 6)
)
ax.barh(
    segment_chart[
        "business_segment"
    ],
    segment_chart[
        "total_revenue"
    ]
)
ax.set_title(
    "Revenue by Customer Segment"
)
ax.set_xlabel(
    "Revenue (£)"
)
fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "customer_segment_revenue.png",
    dpi=200
)
plt.close(
    fig
)
# ============================================================
# 22. REVENUE CONCENTRATION FIGURE
# ============================================================
fig, ax = plt.subplots(
    figsize=(10, 6)
)
ax.plot(
    customer_revenue[
        "customer_share"
    ] * 100,
    customer_revenue[
        "cumulative_revenue_share"
    ] * 100
)
ax.axvline(
    10,
    linestyle="--"
)
ax.set_title(
    "Customer Revenue Concentration"
)
ax.set_xlabel(
    "Cumulative Customer Share (%)"
)
ax.set_ylabel(
    "Cumulative Revenue Share (%)"
)
ax.grid(
    alpha=0.25
)
fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "customer_revenue_concentration.png",
    dpi=200
)
plt.close(
    fig
)
# ============================================================
# 23. K-MEANS SEGMENT FIGURE
# ============================================================
fig, ax = plt.subplots(
    figsize=(10, 6)
)
scatter = ax.scatter(
    customer_rfm[
        "frequency"
    ],
    customer_rfm[
        "monetary"
    ],
    c=customer_rfm[
        "kmeans_cluster"
    ],
    alpha=0.6
)
ax.set_title(
    "Customer Behavioural Segmentation"
)
ax.set_xlabel(
    "Purchase Frequency"
)
ax.set_ylabel(
    "Customer Revenue (£)"
)
fig.colorbar(
    scatter,
    ax=ax,
    label="K-Means Cluster"
)
fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "customer_kmeans_segments.png",
    dpi=200
)
plt.close(
    fig
)
# ============================================================
# 24. FINAL PROJECT SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL PROJECT SUMMARY")
print("=" * 70)
print("\nValid revenue:")
print(
    "£",
    round(
        total_revenue,
        2
    )
)
print("\nCustomers:")
print(total_customers)
print("\nOrders:")
print(total_orders)
print("\nAverage order value:")
print(
    "£",
    round(
        average_order_value,
        2
    )
)
print("\nRepeat-customer rate:")
print(
    round(
        repeat_customer_rate * 100,
        2
    ),
    "%"
)
print("\nInvoice cancellation rate:")
print(
    round(
        cancellation_rate * 100,
        2
    ),
    "%"
)
print(
    "\nRevenue share generated by top 10% of customers:"
)
print(
    round(
        top_10_customer_revenue_share * 100,
        2
    ),
    "%"
)
print("\nSelected K-Means cluster count:")
print(best_k)
print("\nFiles created:")
print(
    "- reports/tables/business_kpis.csv"
)
print(
    "- reports/tables/monthly_kpis.csv"
)
print(
    "- reports/tables/country_performance.csv"
)
print(
    "- reports/tables/product_performance.csv"
)
print(
    "- reports/tables/customer_rfm_segments.csv"
)
print(
    "- reports/tables/customer_cluster_profile.csv"
)
print(
    "- reports/tables/business_segment_summary.csv"
)
print(
    "- reports/tables/cohort_retention.csv"
)
print(
    "- reports/tables/customer_revenue_concentration.csv"
)
print(
    "- reports/tables/silhouette_scores.csv"
)
print(
    "- reports/figures/monthly_revenue.png"
)
print(
    "- reports/figures/cohort_retention_heatmap.png"
)
print(
    "- reports/figures/customer_segment_revenue.png"
)
print(
    "- reports/figures/customer_revenue_concentration.png"
)
print(
    "- reports/figures/customer_kmeans_segments.png"
)
con.close()
print("\n" + "=" * 70)
print("CUSTOMER ANALYTICS PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)