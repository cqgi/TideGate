# Semantic Cache Calibration

pairs=555 positives=250 negatives=305 dataset_sha256=9a8bbeb93e2fb1d942b15f922aa95bdeb81ebaf87c23b35d9101e13826458abd
embedding_model=BAAI/bge-small-zh-v1.5
reranker_model=BAAI/bge-reranker-base
recall_threshold=0.654424 target_recall=0.950

## Two-stage comparison

| method | point | recall_threshold | tau | expected_recall | expected_fpr |
|---|---|---:|---:|---:|---:|
| single-stage bi-encoder | conservative | 0.000000 | 0.955 | 0.016000 | 0.003279 |
| single-stage bi-encoder | balanced | 0.000000 | 0.925 | 0.028000 | 0.019672 |
| single-stage bi-encoder | aggressive | 0.000000 | 0.885 | 0.096000 | 0.045902 |
| recall+rerank | conservative | 0.654424 | 6.800 | 0.088000 | 0.009836 |
| recall+rerank | balanced | 0.654424 | 5.500 | 0.288000 | 0.029508 |
| recall+rerank | aggressive | 0.654424 | 5.200 | 0.352000 | 0.042623 |

