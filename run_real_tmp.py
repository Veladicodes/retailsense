import time, json, logging
logging.basicConfig(level=logging.INFO)
from retailsense.config import load_config
from retailsense.pipeline import tasks
cfg = load_config()
t=time.time()
print(tasks.ingest(cfg, "data/raw/online_retail_ii.csv"), round(time.time()-t))
print(tasks.validate(cfg)); t=time.time()
print(tasks.build_features_task(cfg), round(time.time()-t)); t=time.time()
r = tasks.train(cfg); print(json.dumps(r["summary"], indent=1), round(time.time()-t))
print(tasks.forecast_next_week(cfg))
