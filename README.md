# G223_MP144 - Fuel Station Daily Sales (Patna)

## Group & Problem Details
- **Group ID**: G223
- **Problem ID**: MP144 (Fuel Station Daily Sales - Patna)
- **Batch**: 23
- **Faculty**: Upasna Joshi

### Group Members
- S24CSEU1354 - Akansha Bhutani
- S24CSEU1345 - Parth Rawat
- S24CSEU1766 - Atul Chauhan

---

## Project Overview
This repository contains the end-to-end MLOps pipeline for predicting daily litres sold at fuel stations around Patna for the Patna Fuel Retailers' Federation.

### Pipeline Components
1. **Data Versioning (DVC)**: Raw training and operational data versioned using DVC pointers.
2. **Data Cleaning & Quality (`src/mp144/data.py`)**: Automated detection of 7 data quality issues, correction of target typos, standardizing category variants, and dropping data leakage columns.
3. **Training & Registry (`src/mp144/pipeline.py`)**: Scikit-learn Pipeline tracked in local SQLite MLflow (`mlflow.db`) with `data_sha256` logging, candidate model comparison, and champion model aliasing (`@champion`).
4. **Prediction API (`app/main.py`)**: FastAPI service exposing `GET /health` and `POST /predict` with schema-level validation constraints.
5. **Testing (`tests/test_project.py`)**: Pytest test suite with TestClient integration tests and zero dependency on uncommitted raw CSVs.
