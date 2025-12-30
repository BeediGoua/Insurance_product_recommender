import json
from src.config import DATA_DIR, ARTIFACTS_DIR, TRAIN_PATH, TEST_PATH, BASELINE_VERSION, EVAL_SEED, MIN_BASKET_SIZE_EVAL
from src.pipelines.baseline_pipeline import fit_baseline_from_csv, evaluate_baseline

OUT_DIR = ARTIFACTS_DIR / BASELINE_VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Running Baseline Pipeline...")
print(f"Data: {DATA_DIR}")
print(f"Artifacts: {OUT_DIR}")

# 1. Fit
train_c, test_c, artifact = fit_baseline_from_csv(TRAIN_PATH, TEST_PATH)
artifact.save(OUT_DIR)
# Save cleaned data for App Inspection
train_c.to_parquet(OUT_DIR / "train_cleaned.parquet")
print("Baseline fitted and saved.")

# 2. Evaluate
res = evaluate_baseline(train_c, artifact, seed=EVAL_SEED, min_basket_size=MIN_BASKET_SIZE_EVAL)
metrics = res["metrics"]

print("METRICS:", metrics)
print("\nBY_BASKET_SIZE:\n", res["by_basket_size"].head(10))
print("\nBY_MASKED_PRODUCT:\n", res["by_masked_product"].head(10))

# 3. Save Metrics & Reports
metrics_path = OUT_DIR / "metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2))
print(f"Metrics saved to {metrics_path}")

res["by_basket_size"].to_csv(OUT_DIR / "by_basket_size.csv")
res["by_masked_product"].to_csv(OUT_DIR / "by_masked_product.csv")

if "granular" in res:
    res["granular"].to_parquet(OUT_DIR / "granular_metrics.parquet")
    print(f"Granular metrics saved to {OUT_DIR / 'granular_metrics.parquet'}")

print(f"Detailed reports saved to {OUT_DIR}")
