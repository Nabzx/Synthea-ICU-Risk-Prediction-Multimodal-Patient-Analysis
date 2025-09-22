Multimodal ICU Risk Prediction with BigQuery AI

## Overview
This project combines vector search, AI-generated insights, and patient lab features to predict ICU admission risk from Synthea patient data. It leverages BigQuery ML, GenAI, and nearest-neighbor analytics for a fully integrated pipeline.

## Features
- Vector embeddings for patient notes
- AI-generated ICU risk predictions (boolean and numeric)
- Clinical summaries from AI
- Neighbor-based lab aggregation and rare-pattern detection
- Alerts table for high-risk patients
- Ready-to-export results for Kaggle competitions

## Setup
1. Replace `PROJECT`, `DATASET`, and `CONNECTION_ID` in SQL queries.
2. Ensure embeddings are stored as `ARRAY<FLOAT64>` in BigQuery.
3. Optional: Enable AI features with a BigQuery Connection.

## Usage
- Run the SQL pipeline in BigQuery UI to generate multimodal patient risk insights.
- Export `patient_alerts` for downstream analysis or Kaggle submissions.

## Files
- `multimodal_pipeline.sql` — main BigQuery SQL pipeline.
- `patient_preprocessing.ipynb` — Colab notebook for data preprocessing.
- `patient_alerts_export.csv` — sample output (if run).

## Notes
- This project demonstrates combining structured and unstructured EHR data.
- Designed for the Google Kaggle competition and clinical AI research.

