# Harness Notes

`battery_soh_rul_anomaly.json` is a draft SE-Bench task definition.

For local VM testing, serve assets from:

```text
http://172.17.0.1:8000/battery_soh_rul/battery_work.tar.gz
http://172.17.0.1:8000/battery_soh_rul/battery_judge.tar.gz
```

For final submission, replace these with stable project-hosted URLs or provide `BATTERY_WORK_ASSET_URL` and `BATTERY_JUDGE_ASSET_URL` at build time. Do not publish the judge asset publicly unless hidden labels are allowed to be public.
