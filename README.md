# Customer Revenue, Retention & Segmentation Analytics

An end-to-end customer and product analytics project using more than
500,000 real retail transaction records to analyse revenue, customer
retention, purchasing behaviour, customer value and behavioural
segmentation.

The project combines Python, SQL, DuckDB, cohort analysis, RFM customer
analytics and unsupervised machine learning to translate transaction-level
data into commercially useful customer insights.

---

## Business Problem

Retail businesses need to understand not only how much revenue they
generate, but also:

- Which customers create the most value?
- How concentrated is revenue across the customer base?
- Which customers purchase repeatedly?
- Which customer groups are loyal, high-value or at risk?
- How does customer retention evolve after acquisition?
- Which products contribute the most revenue?
- Can customer behaviour be separated into meaningful data-driven segments?

This project builds an analytical pipeline to answer these questions.

---

## Dataset

The project uses the **UCI Online Retail dataset**, containing transaction
data for a UK-based non-store retailer.

The raw dataset contains:

**541,909 transaction rows**

covering:

**1 December 2010 to 9 December 2011**

Main variables include:

- Invoice number
- Product code
- Product description
- Quantity
- Invoice date
- Unit price
- Customer ID
- Country

Dataset attribution:

Daqing Chen. Online Retail.  
UCI Machine Learning Repository.  
Dataset ID 352.  
DOI: 10.24432/C5BW33

---

## Technology Stack

- Python
- SQL
- DuckDB
- pandas
- NumPy
- scikit-learn
- matplotlib
- UCI Machine Learning Repository
- Git / GitHub

---

## Analytical Pipeline

```text
UCI Online Retail Data
        ↓
Data ingestion
        ↓
Data-quality audit
        ↓
Duplicate removal
        ↓
Cancellation / invalid transaction handling
        ↓
Transaction-level dataset
        ↓
Order-level dataset
        ↓
DuckDB + SQL analytics
        ↓
Business KPI analysis
        ↓
Product and geographic analytics
        ↓
Customer RFM analysis
        ↓
K-Means behavioural segmentation
        ↓
Cohort retention analysis
        ↓
Customer revenue concentration
        ↓
Business interpretation