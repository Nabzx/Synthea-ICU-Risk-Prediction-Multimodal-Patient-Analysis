-- ============================================================================
-- FINAL: Multimodal Pipeline + Advanced Features (vector search, alerts, rare-pattern analytics, evaluation, BQML XGBoost)
-- Project: big-query-hackathon-472816
-- Dataset: synthea_demo
-- GenAI Connection: projects/big-query-hackathon-472816/locations/us/connections/my-genai-conn
-- Notes: Run in US region. AI blocks are commented out; enable when billing + connection available.
-- ============================================================================

-- --------------------------------------------------------------------------
-- 0) Ensure embeddings column is ARRAY<FLOAT64>
--    Creates patient_embeddings_vectorized from patient_embeddings if needed.
-- --------------------------------------------------------------------------
CREATE OR REPLACE TABLE `big-query-hackathon-472816.synthea_demo.patient_embeddings_vectorized` AS
SELECT
  patient_id,
  note,
  CAST(embedding AS ARRAY<FLOAT64>) AS embedding
FROM `big-query-hackathon-472816.synthea_demo.patient_embeddings`;

-- --------------------------------------------------------------------------
-- 1) Core multimodal pipeline (no-AI version — runnable now)
--    - Use a sample embedding from an existing patient as "new note"
--    - VECTOR_SEARCH to get top-K neighbors
--    - Aggregate neighbor lab statistics (safe casting)
-- --------------------------------------------------------------------------
WITH
new_note AS (
  SELECT embedding
  FROM `big-query-hackathon-472816.synthea_demo.patient_embeddings_vectorized`
  WHERE patient_id = 'some_existing_patient_id'  -- <-- replace with a real patient_id
  LIMIT 1
),

vector_match AS (
  SELECT
    base.id AS patient_id,
    base.note AS combined_text,
    distance
  FROM VECTOR_SEARCH(
    TABLE `big-query-hackathon-472816.synthea_demo.patient_embeddings_vectorized`,
    'embedding',
    (SELECT embedding FROM new_note),
    top_k => 10,
    distance_type => 'COSINE'
  ) AS base
),

neighbor_features AS (
  SELECT
    ARRAY_AGG(vm.patient_id) AS neighbor_ids,
    AVG(SAFE_CAST(f.WBC AS FLOAT64)) AS neighbor_WBC_avg,
    AVG(SAFE_CAST(f.Hemoglobin AS FLOAT64)) AS neighbor_Hb_avg,
    AVG(SAFE_CAST(f.Creatinine AS FLOAT64)) AS neighbor_Creatinine_avg,
    COUNT(1) AS neighbor_count
  FROM vector_match vm
  LEFT JOIN `big-query-hackathon-472816.synthea_demo.patient_features` f
    ON vm.patient_id = f.patient_id
)

SELECT
  vm.patient_id,
  vm.combined_text,
  SAFE_CAST(f.WBC AS FLOAT64)       AS WBC,
  SAFE_CAST(f.Hemoglobin AS FLOAT64) AS Hemoglobin,
  SAFE_CAST(f.Creatinine AS FLOAT64) AS Creatinine,
  nf.neighbor_WBC_avg,
  nf.neighbor_Hb_avg,
  nf.neighbor_Creatinine_avg,
  nf.neighbor_count,
  vm.distance
FROM vector_match vm
LEFT JOIN `big-query-hackathon-472816.synthea_demo.patient_features` f
  ON vm.patient_id = f.patient_id
CROSS JOIN neighbor_features nf
ORDER BY vm.distance ASC;

-- --------------------------------------------------------------------------
-- 2) Advanced analytics: nearest-neighbor summary & rare-pattern detection
-- --------------------------------------------------------------------------
-- 2a) Nearest-neighbor distance distribution for the top-K neighbors
WITH
nn_query AS (
  SELECT embedding FROM `big-query-hackathon-472816.synthea_demo.patient_embeddings_vectorized`
  WHERE patient_id = 'some_existing_patient_id' LIMIT 1
),
vm AS (
  SELECT base.patient_id, base.distance
  FROM VECTOR_SEARCH(
    TABLE `big-query-hackathon-472816.synthea_demo.patient_embeddings_vectorized`,
    'embedding',
    (SELECT embedding FROM nn_query),
    top_k => 100,
    distance_type => 'COSINE'
  ) AS base
)
SELECT
  COUNT(*) AS neighbors_considered,
  MIN(distance) AS min_distance,
  APPROX_QUANTILES(distance, 5) AS quintile_distances,
  MAX(distance) AS max_distance
FROM vm;

-- 2b) Find clusters around rare lab patterns (example: creatinine > 2.0)
WITH anchors AS (
  SELECT patient_id, embedding
  FROM `big-query-hackathon-472816.synthea_demo.patient_features`
  WHERE SAFE_CAST(Creatinine AS FLOAT64) > 2.0
  LIMIT 5
),
neighbors AS (
  SELECT a.patient_id AS anchor_id, vm.patient_id AS neighbor_id, vm.distance
  FROM anchors a,
       VECTOR_SEARCH(
         TABLE `big-query-hackathon-472816.synthea_demo.patient_embeddings_vectorized`,
         'embedding',
         a.embedding,
         top_k => 50,
         distance_type => 'COSINE'
       ) vm
)
SELECT anchor_id, COUNT(DISTINCT neighbor_id) AS neighbor_count, MIN(distance) AS closest
FROM neighbors
GROUP BY anchor_id
ORDER BY neighbor_count DESC;

-- --------------------------------------------------------------------------
-- 3) Alerts table creation (persist results for downstream integration)
--    Creates/overwrites patient_alerts table with a heuristic risk score.
-- --------------------------------------------------------------------------
CREATE OR REPLACE TABLE `big-query-hackathon-472816.synthea_demo.patient_alerts` AS
WITH
sample_queries AS (
  SELECT patient_id, embedding FROM `big-query-hackathon-472816.synthea_demo.patient_embeddings_vectorized` LIMIT 50
),
matched AS (
  SELECT
    q.patient_id AS query_id,
    base.patient_id AS neighbor_id,
    base.distance
  FROM sample_queries q,
       VECTOR_SEARCH(
         TABLE `big-query-hackathon-472816.synthea_demo.patient_embeddings_vectorized`,
         'embedding',
         q.embedding,
         top_k => 10,
         distance_type => 'COSINE'
       ) AS base
)
SELECT
  m.query_id AS alert_for_patient,
  m.neighbor_id,
  m.distance,
  SAFE_CAST(pf.WBC AS FLOAT64) AS WBC,
  SAFE_CAST(pf.Hemoglobin AS FLOAT64) AS Hemoglobin,
  SAFE_CAST(pf.Creatinine AS FLOAT64) AS Creatinine,
  -- heuristic risk score: normalized creatinine + normalized WBC + inverse distance
  (COALESCE(SAFE_CAST(pf.Creatinine AS FLOAT64),0)/NULLIF((SELECT MAX(SAFE_CAST(Creatinine AS FLOAT64)) FROM `big-query-hackathon-472816.synthea_demo.patient_features`),0)
   + COALESCE(SAFE_CAST(pf.WBC AS FLOAT64),0)/NULLIF((SELECT MAX(SAFE_CAST(WBC AS FLOAT64)) FROM `big-query-hackathon-472816.synthea_demo.patient_features`),0)
   + (1 - m.distance)
  ) / 3.0 AS heuristic_risk_score
FROM matched m
LEFT JOIN `big-query-hackathon-472816.synthea_demo.patient_features` pf
  ON m.neighbor_id = pf.patient_id
WHERE (COALESCE(SAFE_CAST(pf.Creatinine AS FLOAT64),0) > 2.0 OR COALESCE(SAFE_CAST(pf.WBC AS FLOAT64),0) > 12.0)
ORDER BY heuristic_risk_score DESC
LIMIT 500;

-- --------------------------------------------------------------------------
-- 4) FULL multimodal pipeline WITH AI (commented out)
--    Enable when you have billing + a GenAI Connection named:
--    projects/big-query-hackathon-472816/locations/us/connections/my-genai-conn
-- --------------------------------------------------------------------------
/*
WITH
-- generate embedding for an arbitrary new note (requires ML.GENERATE_EMBEDDING)
new_note AS (
  SELECT text_embedding AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `ml-bqml-public.mlglobal_text_embedding_gecko_003`,
    (SELECT 'Patient reports cough, fever and elevated creatinine.' AS content)
  )
),

-- vector search using the generated embedding
vector_match AS (
  SELECT base.patient_id, base.note AS combined_text, distance
  FROM VECTOR_SEARCH(
    TABLE `big-query-hackathon-472816.synthea_demo.patient_embeddings_vectorized`,
    'embedding',
    (SELECT embedding FROM new_note),
    top_k => 10,
    distance_type => 'COSINE'
  ) AS base
),

neighbor_features AS (
  SELECT
    ARRAY_AGG(vm.patient_id) AS neighbor_ids,
    AVG(SAFE_CAST(f.WBC AS FLOAT64)) AS neighbor_WBC_avg,
    AVG(SAFE_CAST(f.Hemoglobin AS FLOAT64)) AS neighbor_Hb_avg,
    AVG(SAFE_CAST(f.Creatinine AS FLOAT64)) AS neighbor_Creatinine_avg
  FROM vector_match vm
  LEFT JOIN `big-query-hackathon-472816.synthea_demo.patient_features` f USING(patient_id)
)

SELECT
  vm.patient_id,
  vm.combined_text,
  SAFE_CAST(f.WBC AS FLOAT64) AS WBC,
  SAFE_CAST(f.Hemoglobin AS FLOAT64) AS Hemoglobin,
  SAFE_CAST(f.Creatinine AS FLOAT64) AS Creatinine,
  nf.neighbor_WBC_avg,
  nf.neighbor_Hb_avg,
  nf.neighbor_Creatinine_avg,
  -- AI-generated boolean: needs ICU (requires BigQuery connection to Vertex/GenAI)
  AI.GENERATE_BOOL(
    model => 'gemini-pro',
    prompt => CONCAT(
      'Based on these notes and labs, does this patient need ICU? ',
      'Answer true or false. Notes: ', vm.combined_text,
      ' Labs: WBC=', CAST(f.WBC AS STRING),
      ', Hemoglobin=', CAST(f.Hemoglobin AS STRING),
      ', Creatinine=', CAST(f.Creatinine AS STRING)
    ),
    temperature => 0.0,
    connection_id => 'projects/big-query-hackathon-472816/locations/us/connections/my-genai-conn'
  ) AS needs_icu_bool,
  -- AI-generated numeric risk
  AI.GENERATE_DOUBLE(
    model => 'gemini-pro',
    prompt => CONCAT(
      'Predict ICU-risk score 0.0-1.0. Notes: ', vm.combined_text,
      '. Labs: WBC=', CAST(f.WBC AS STRING),
      ', Creatinine=', CAST(f.Creatinine AS STRING)
    ),
    temperature => 0.0,
    connection_id => 'projects/big-query-hackathon-472816/locations/us/connections/my-genai-conn'
  ) AS ai_icu_risk_score,
  AI.GENERATE_TEXT(
    model => 'gemini-pro',
    prompt => CONCAT(
      'Summarize patient notes and recommend next steps. Notes: ', vm.combined_text,
      '. Labs: WBC=', CAST(f.WBC AS STRING),
      ', Hemoglobin=', CAST(f.Hemoglobin AS STRING),
      ', Creatinine=', CAST(f.Creatinine AS STRING)
    ),
    temperature => 0.2,
    max_output_tokens => 256,
    connection_id => 'projects/big-query-hackathon-472816/locations/us/connections/my-genai-conn'
  ) AS clinical_summary,
  vm.distance
FROM vector_match vm
LEFT JOIN `big-query-hackathon-472816.synthea_demo.patient_features` f USING(patient_id)
CROSS JOIN neighbor_features nf
ORDER BY vm.distance ASC;
*/

-- --------------------------------------------------------------------------
-- 5) Evaluation / Metrics
--    Compare model (BQML) predictions to ground-truth icu_admit where available.
-- --------------------------------------------------------------------------
-- 5a) Train a BQML boosted-tree model (structured baseline). If icu_admit missing, this will error; create a synthetic label if needed.
CREATE OR REPLACE MODEL `big-query-hackathon-472816.synthea_demo.icu_xgb`
OPTIONS(
  model_type='boosted_tree_classifier',
  input_label_cols=['icu_admit'],
  max_iterations=50
) AS
SELECT
  SAFE_CAST(WBC AS FLOAT64) AS WBC,
  SAFE_CAST(Hemoglobin AS FLOAT64) AS Hemoglobin,
  SAFE_CAST(Creatinine AS FLOAT64) AS Creatinine,
  icu_admit
FROM `big-query-hackathon-472816.synthea_demo.patient_features`
WHERE icu_admit IS NOT NULL;

-- 5b) Evaluate the trained model
SELECT * FROM ML.EVALUATE(MODEL `big-query-hackathon-472816.synthea_demo.icu_xgb`, (
  SELECT SAFE_CAST(WBC AS FLOAT64) AS WBC, SAFE_CAST(Hemoglobin AS FLOAT64) AS Hemoglobin, SAFE_CAST(Creatinine AS FLOAT64) AS Creatinine, icu_admit
  FROM `big-query-hackathon-472816.synthea_demo.patient_features`
  WHERE icu_admit IS NOT NULL
));

-- 5c) Predict probabilities and save into a table for downstream joins (overwrite each run)
CREATE OR REPLACE TABLE `big-query-hackathon-472816.synthea_demo.patient_xgb_preds` AS
SELECT
  patient_id,
  predicted_icu_admit_probs[OFFSET(1)] AS xgb_pred_prob
FROM ML.PREDICT(MODEL `big-query-hackathon-472816.synthea_demo.icu_xgb`,
  (
    SELECT patient_id, SAFE_CAST(WBC AS FLOAT64) AS WBC, SAFE_CAST(Hemoglobin AS FLOAT64) AS Hemoglobin, SAFE_CAST(Creatinine AS FLOAT64) AS Creatinine
    FROM `big-query-hackathon-472816.synthea_demo.patient_features`
  )
);

-- 5d) Quick evaluation join: compare XGBoost predictions to ground-truth where available (threshold 0.5)
SELECT
  COUNT(*) AS total_eval,
  SUM(CASE WHEN CAST(pf.icu_admit AS INT64) = 1 AND x.xgb_pred_prob >= 0.5 THEN 1 ELSE 0 END) AS true_positives,
  SUM(CASE WHEN CAST(pf.icu_admit AS INT64) = 1 AND x.xgb_pred_prob < 0.5 THEN 1 ELSE 0 END) AS false_negatives,
  SUM(CASE WHEN CAST(pf.icu_admit AS INT64) = 0 AND x.xgb_pred_prob >= 0.5 THEN 1 ELSE 0 END) AS false_positives,
  SUM(CASE WHEN CAST(pf.icu_admit AS INT64) = 0 AND x.xgb_pred_prob < 0.5 THEN 1 ELSE 0 END) AS true_negatives
FROM `big-query-hackathon-472816.synthea_demo.patient_features` pf
JOIN `big-query-hackathon-472816.synthea_demo.patient_xgb_preds` x
  ON pf.patient_id = x.patient_id
WHERE pf.icu_admit IS NOT NULL;

-- --------------------------------------------------------------------------
-- 6) Optional: Export patient_alerts to GCS for Kaggle or dashboarding (commented)
-- --------------------------------------------------------------------------
/*
EXPORT DATA OPTIONS(
  uri='gs://YOUR_BUCKET/final_patient_alerts_*.csv',
  format='CSV',
  overwrite=true
) AS
SELECT * FROM `big-query-hackathon-472816.synthea_demo.patient_alerts`;
*/

-- ============================================================================
-- End of script
-- ============================================================================
