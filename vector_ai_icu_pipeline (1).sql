-- ============================================================================
-- FINAL: Multimodal Pipeline + Advanced Features (vector search, alerts, rare-pattern analytics, evaluation, BQML XGBoost)
-- Project: hackathon-472921
-- Dataset: synthea_demo
-- GenAI Connection: projects/hackathon-472921/locations/us/connections/my-genai-conn
-- Notes: Run in US region. AI blocks are DISABLED for demo; commented out below.
-- ============================================================================

-- --------------------------------------------------------------------------
-- 0) Ensure embeddings column is ARRAY<FLOAT64>
--    Creates patient_embeddings_vectorized from patient_embeddings if needed.
-- --------------------------------------------------------------------------
CREATE OR REPLACE TABLE `hackathon-472921.synthea_demo.patient_embeddings_vectorized` AS
SELECT
  patient_id,
  note,
  CAST(embedding AS ARRAY<FLOAT64>) AS embedding
FROM `hackathon-472921.synthea_demo.patient_embeddings`;

-- --------------------------------------------------------------------------
-- 1) Core multimodal pipeline (no-AI version — runnable now)
--    - Use a sample embedding from an existing patient as "new note"
--    - VECTOR_SEARCH to get top-K neighbors
--    - Aggregate neighbor lab statistics (safe casting)
-- --------------------------------------------------------------------------
WITH
new_note AS (
  SELECT embedding
  FROM `hackathon-472921.synthea_demo.patient_embeddings_vectorized`
  WHERE patient_id IN (
    SELECT ANY_VALUE(patient_id)
    FROM `hackathon-472921.synthea_demo.patient_embeddings_vectorized`
    LIMIT 1
  )
  LIMIT 1
),

vector_match AS (
  SELECT
    vs.base.patient_id AS patient_id,
    vs.base.note AS combined_text,
    vs.distance AS distance
  FROM VECTOR_SEARCH(
    TABLE `hackathon-472921.synthea_demo.patient_embeddings_vectorized`,
    'embedding',
    (SELECT embedding FROM new_note),
    top_k => 10,
    distance_type => 'COSINE'
  ) AS vs
),

neighbor_features AS (
  SELECT
    ARRAY_AGG(vm.patient_id) AS neighbor_ids,
    AVG(SAFE_CAST(f.WBC AS FLOAT64)) AS neighbor_WBC_avg,
    AVG(SAFE_CAST(f.Hemoglobin AS FLOAT64)) AS neighbor_Hb_avg,
    AVG(SAFE_CAST(f.Creatinine AS FLOAT64)) AS neighbor_Creatinine_avg,
    COUNT(1) AS neighbor_count
  FROM vector_match vm
  LEFT JOIN `hackathon-472921.synthea_demo.patient_features` f
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
LEFT JOIN `hackathon-472921.synthea_demo.patient_features` f
  ON vm.patient_id = f.patient_id
CROSS JOIN neighbor_features nf
ORDER BY vm.distance ASC;

-- --------------------------------------------------------------------------
-- 2) Advanced analytics: nearest-neighbor summary & rare-pattern detection
-- --------------------------------------------------------------------------
-- 2a) Nearest-neighbor distance distribution for the top-K neighbors
WITH
nn_query AS (
SELECT embedding FROM `hackathon-472921.synthea_demo.patient_embeddings_vectorized`
  ORDER BY RAND() LIMIT 1
),
vm AS (
  SELECT vs.base.patient_id AS patient_id, vs.distance AS distance
  FROM VECTOR_SEARCH(
    TABLE `hackathon-472921.synthea_demo.patient_embeddings_vectorized`,
    'embedding',
    (SELECT embedding FROM nn_query),
    top_k => 100,
    distance_type => 'COSINE'
  ) AS vs
)
SELECT
  COUNT(*) AS neighbors_considered,
  MIN(distance) AS min_distance,
  APPROX_QUANTILES(distance, 5) AS quintile_distances,
  MAX(distance) AS max_distance
FROM vm;

-- 2b) Find clusters around rare lab patterns (example: creatinine > 2.0)
WITH anchors AS (
  SELECT pf.patient_id, pev.embedding
  FROM `hackathon-472921.synthea_demo.patient_features` pf
  JOIN `hackathon-472921.synthea_demo.patient_embeddings_vectorized` pev
    ON pf.patient_id = pev.patient_id
  WHERE SAFE_CAST(pf.Creatinine AS FLOAT64) > 2.0
  LIMIT 5
),
neighbors AS (
  SELECT
    vs.query.patient_id AS anchor_id,
    vs.base.patient_id AS neighbor_id,
    vs.distance AS distance
  FROM VECTOR_SEARCH(
    TABLE `hackathon-472921.synthea_demo.patient_embeddings_vectorized`,
    'embedding',
    TABLE anchors,
    'embedding',
    top_k => 50,
    distance_type => 'COSINE'
  ) AS vs
)
SELECT anchor_id, COUNT(DISTINCT neighbor_id) AS neighbor_count, MIN(distance) AS closest
FROM neighbors
GROUP BY anchor_id
ORDER BY neighbor_count DESC;

-- --------------------------------------------------------------------------
-- 3) Alerts table creation (persist results for downstream integration)
--    Creates/overwrites patient_alerts table with a heuristic risk score.
-- --------------------------------------------------------------------------
CREATE OR REPLACE TABLE `hackathon-472921.synthea_demo.patient_alerts` AS
WITH
sample_queries AS (
  SELECT patient_id, embedding FROM `hackathon-472921.synthea_demo.patient_embeddings_vectorized` LIMIT 200
),
matched AS (
  SELECT
    vs.query.patient_id AS patient_id,
    vs.base.patient_id AS neighbor_id,
    vs.distance AS distance
  FROM VECTOR_SEARCH(
    TABLE `hackathon-472921.synthea_demo.patient_embeddings_vectorized`,
    'embedding',
    TABLE sample_queries,
    'embedding',
    top_k => 10,
    distance_type => 'COSINE'
  ) AS vs
),
scored AS (
  SELECT
    m.patient_id,
    -- heuristic risk score: normalized creatinine + normalized WBC + inverse distance
    (COALESCE(SAFE_CAST(pf.Creatinine AS FLOAT64),0)/NULLIF((SELECT MAX(SAFE_CAST(Creatinine AS FLOAT64)) FROM `hackathon-472921.synthea_demo.patient_features`),0)
     + COALESCE(SAFE_CAST(pf.WBC AS FLOAT64),0)/NULLIF((SELECT MAX(SAFE_CAST(WBC AS FLOAT64)) FROM `hackathon-472921.synthea_demo.patient_features`),0)
     + (1 - m.distance)
    ) / 3.0 AS heuristic_component
  FROM matched m
  LEFT JOIN `hackathon-472921.synthea_demo.patient_features` pf
    ON m.neighbor_id = pf.patient_id
),
agg AS (
  SELECT
    patient_id,
    AVG(heuristic_component) AS heuristic_risk_score
  FROM scored
  GROUP BY patient_id
)
SELECT
  a.patient_id,
  -- binary flag with threshold = 0.5 as per context
  a.heuristic_risk_score >= 0.5 AS icu_flag,
  -- use heuristic as the continuous score when AI is disabled
  a.heuristic_risk_score AS icu_risk_score,
  a.heuristic_risk_score AS heuristic_risk_score,
  'Neighbor evidence supports elevated risk' AS summary,
  CURRENT_TIMESTAMP() AS ts
FROM agg a
ORDER BY icu_risk_score DESC
LIMIT 500;

-- --------------------------------------------------------------------------
-- 4) FULL multimodal pipeline WITH AI (ENABLED)
--    Requires GenAI Connection:
--    projects/hackathon-472921/locations/us/connections/my-genai-conn
-- --------------------------------------------------------------------------
/*
WITH
-- use an existing patient embedding as the query vector (avoids external model dependency)
new_note AS (
  SELECT embedding
  FROM `hackathon-472921.synthea_demo.patient_embeddings_vectorized`
  ORDER BY RAND()
  LIMIT 1
),

-- vector search using the generated embedding
vector_match AS (
  SELECT vs.base.patient_id AS patient_id, vs.base.note AS combined_text, vs.distance
  FROM VECTOR_SEARCH(
    TABLE `hackathon-472921.synthea_demo.patient_embeddings_vectorized`,
    'embedding',
    (SELECT embedding FROM new_note),
    top_k => 10,
    distance_type => 'COSINE'
  ) AS vs
),

neighbor_features AS (
  SELECT
    ARRAY_AGG(vm.patient_id) AS neighbor_ids,
    AVG(SAFE_CAST(f.WBC AS FLOAT64)) AS neighbor_WBC_avg,
    AVG(SAFE_CAST(f.Hemoglobin AS FLOAT64)) AS neighbor_Hb_avg,
    AVG(SAFE_CAST(f.Creatinine AS FLOAT64)) AS neighbor_Creatinine_avg
  FROM vector_match vm
  LEFT JOIN `hackathon-472921.synthea_demo.patient_features` f USING(patient_id)
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
    prompt => CONCAT(
      'Based on these notes and labs, does this patient need ICU? ',
      'Answer true or false. Notes: ', vm.combined_text,
      ' Labs: WBC=', CAST(f.WBC AS STRING),
      ', Hemoglobin=', CAST(f.Hemoglobin AS STRING),
      ', Creatinine=', CAST(f.Creatinine AS STRING)
    ),
    connection_id => 'hackathon-472921.us.my-genai-conn',
    model_params => JSON '{"temperature": 0.0}'
  ).result AS needs_icu_bool,
  -- AI-generated numeric risk
  AI.GENERATE_DOUBLE(
    prompt => CONCAT(
      'Predict ICU-risk score 0.0-1.0. Notes: ', vm.combined_text,
      '. Labs: WBC=', CAST(f.WBC AS STRING),
      ', Creatinine=', CAST(f.Creatinine AS STRING)
    ),
    connection_id => 'hackathon-472921.us.my-genai-conn',
    model_params => JSON '{"temperature": 0.0}'
  ).result AS ai_icu_risk_score,
  ai.generate_text(
    prompt => CONCAT(
      'Summarize patient notes and recommend next steps. Notes: ', vm.combined_text,
      '. Labs: WBC=', CAST(f.WBC AS STRING),
      ', Hemoglobin=', CAST(f.Hemoglobin AS STRING),
      ', Creatinine=', CAST(f.Creatinine AS STRING)
    ),
    connection_id => 'hackathon-472921.us.my-genai-conn',
    model_params => JSON '{"temperature": 0.2, "max_output_tokens": 256}'
  ) AS clinical_summary,
  vm.distance
FROM vector_match vm
LEFT JOIN `hackathon-472921.synthea_demo.patient_features` f USING(patient_id)
CROSS JOIN neighbor_features nf
ORDER BY vm.distance ASC;
*/

-- --------------------------------------------------------------------------
-- 5) Evaluation / Metrics
--    Compare model (BQML) predictions to ground-truth icu_admit where available.
-- --------------------------------------------------------------------------
-- 5a) Train a BQML boosted-tree model (structured baseline). If icu_admit missing, this will error; create a synthetic label if needed.
CREATE OR REPLACE MODEL `hackathon-472921.synthea_demo.icu_xgb`
OPTIONS(
  model_type='boosted_tree_classifier',
  input_label_cols=['label_bool'],
  max_iterations=50
) AS
SELECT
  COALESCE(
    SAFE_CAST(pf.WBC AS FLOAT64),
    (SELECT AVG(SAFE_CAST(WBC AS FLOAT64)) FROM `hackathon-472921.synthea_demo.patient_features`),
    0.0
  ) AS WBC,
  COALESCE(
    SAFE_CAST(pf.Hemoglobin AS FLOAT64),
    (SELECT AVG(SAFE_CAST(Hemoglobin AS FLOAT64)) FROM `hackathon-472921.synthea_demo.patient_features`),
    0.0
  ) AS Hemoglobin,
  COALESCE(
    SAFE_CAST(pf.Creatinine AS FLOAT64),
    (SELECT AVG(SAFE_CAST(Creatinine AS FLOAT64)) FROM `hackathon-472921.synthea_demo.patient_features`),
    0.0
  ) AS Creatinine,
  CASE
    WHEN (
      SELECT COUNT(DISTINCT CAST(icu_admit AS INT64))
      FROM `hackathon-472921.synthea_demo.patient_features`
      WHERE icu_admit IS NOT NULL
    ) >= 2 THEN COALESCE(
      CAST(pf.icu_admit AS BOOL),
      a.heuristic_risk_score >= 0.5,
      SAFE_CAST(pf.Creatinine AS FLOAT64) > 2.0,
      SAFE_CAST(pf.WBC AS FLOAT64) > 12.0,
      FALSE
    )
    ELSE COALESCE(
      a.heuristic_risk_score >= 0.5,
      SAFE_CAST(pf.Creatinine AS FLOAT64) > 2.0,
      SAFE_CAST(pf.WBC AS FLOAT64) > 12.0,
      FALSE
    )
  END AS label_bool
FROM `hackathon-472921.synthea_demo.patient_features` pf
LEFT JOIN `hackathon-472921.synthea_demo.patient_alerts` a
  ON pf.patient_id = a.patient_id;

-- 5b) Evaluate the trained model
SELECT * FROM ML.EVALUATE(MODEL `hackathon-472921.synthea_demo.icu_xgb`, (
  SELECT
    COALESCE(
      SAFE_CAST(pf.WBC AS FLOAT64),
      (SELECT AVG(SAFE_CAST(WBC AS FLOAT64)) FROM `hackathon-472921.synthea_demo.patient_features`),
      0.0
    ) AS WBC,
    COALESCE(
      SAFE_CAST(pf.Hemoglobin AS FLOAT64),
      (SELECT AVG(SAFE_CAST(Hemoglobin AS FLOAT64)) FROM `hackathon-472921.synthea_demo.patient_features`),
      0.0
    ) AS Hemoglobin,
    COALESCE(
      SAFE_CAST(pf.Creatinine AS FLOAT64),
      (SELECT AVG(SAFE_CAST(Creatinine AS FLOAT64)) FROM `hackathon-472921.synthea_demo.patient_features`),
      0.0
    ) AS Creatinine,
    CASE
      WHEN (
        SELECT COUNT(DISTINCT CAST(icu_admit AS INT64))
        FROM `hackathon-472921.synthea_demo.patient_features`
        WHERE icu_admit IS NOT NULL
      ) >= 2 THEN COALESCE(
        CAST(pf.icu_admit AS BOOL),
        a.heuristic_risk_score >= 0.5,
        SAFE_CAST(pf.Creatinine AS FLOAT64) > 2.0,
        SAFE_CAST(pf.WBC AS FLOAT64) > 12.0,
        FALSE
      )
      ELSE COALESCE(
        a.heuristic_risk_score >= 0.5,
        SAFE_CAST(pf.Creatinine AS FLOAT64) > 2.0,
        SAFE_CAST(pf.WBC AS FLOAT64) > 12.0,
        FALSE
      )
    END AS label_bool
  FROM `hackathon-472921.synthea_demo.patient_features` pf
  LEFT JOIN `hackathon-472921.synthea_demo.patient_alerts` a
    ON pf.patient_id = a.patient_id
));

-- 5c) Predict probabilities and save into a table for downstream joins (overwrite each run)
CREATE OR REPLACE TABLE `hackathon-472921.synthea_demo.patient_xgb_preds` AS
SELECT
  patient_id,
  predicted_icu_admit_probs[OFFSET(1)] AS xgb_pred_prob
FROM ML.PREDICT(MODEL `hackathon-472921.synthea_demo.icu_xgb`,
  (
    SELECT
      patient_id,
      COALESCE(
        SAFE_CAST(WBC AS FLOAT64),
        (SELECT AVG(SAFE_CAST(WBC AS FLOAT64)) FROM `hackathon-472921.synthea_demo.patient_features`),
        0.0
      ) AS WBC,
      COALESCE(
        SAFE_CAST(Hemoglobin AS FLOAT64),
        (SELECT AVG(SAFE_CAST(Hemoglobin AS FLOAT64)) FROM `hackathon-472921.synthea_demo.patient_features`),
        0.0
      ) AS Hemoglobin,
      COALESCE(
        SAFE_CAST(Creatinine AS FLOAT64),
        (SELECT AVG(SAFE_CAST(Creatinine AS FLOAT64)) FROM `hackathon-472921.synthea_demo.patient_features`),
        0.0
      ) AS Creatinine
    FROM `hackathon-472921.synthea_demo.patient_features`
  )
);

-- 5d) Quick evaluation join: compare XGBoost predictions to ground-truth where available (threshold 0.5)
SELECT
  COUNT(*) AS total_eval,
  SUM(CASE WHEN CAST(pf.icu_admit AS INT64) = 1 AND x.xgb_pred_prob >= 0.5 THEN 1 ELSE 0 END) AS true_positives,
  SUM(CASE WHEN CAST(pf.icu_admit AS INT64) = 1 AND x.xgb_pred_prob < 0.5 THEN 1 ELSE 0 END) AS false_negatives,
  SUM(CASE WHEN CAST(pf.icu_admit AS INT64) = 0 AND x.xgb_pred_prob >= 0.5 THEN 1 ELSE 0 END) AS false_positives,
  SUM(CASE WHEN CAST(pf.icu_admit AS INT64) = 0 AND x.xgb_pred_prob < 0.5 THEN 1 ELSE 0 END) AS true_negatives
FROM `hackathon-472921.synthea_demo.patient_features` pf
JOIN `hackathon-472921.synthea_demo.patient_xgb_preds` x
  ON pf.patient_id = x.patient_id
WHERE pf.icu_admit IS NOT NULL;

-- --------------------------------------------------------------------------
-- 6) Optional: Export patient_alerts to GCS for Kaggle or dashboarding (commented)
-- --------------------------------------------------------------------------
/*
EXPORT DATA OPTIONS(
  uri='gs://${BUCKET}/final_patient_alerts_*.csv',
  format='CSV',
  overwrite=true
) AS
SELECT * FROM `hackathon-472921.synthea_demo.patient_alerts`;
*/

-- ============================================================================
-- End of script
-- ============================================================================
