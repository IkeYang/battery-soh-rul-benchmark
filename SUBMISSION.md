# Submission Notes

Owner: IKE1997

Repository URL:
https://github.com/IKE1997/battery-soh-rul-benchmark

Recommended release tag:
`v0.1.0`

Release assets to upload:

| File | Purpose |
|---|---|
| `battery_work.tar.gz` | Agent-visible work payload used by the SE-Bench work image. |
| `battery_judge.tar.gz` | Judge payload containing scorer and hidden eval data. |
| `SHA256SUMS` | Integrity checksums for both payloads. |

Release asset URLs after upload:

```text
https://github.com/IKE1997/battery-soh-rul-benchmark/releases/download/v0.1.0/battery_work.tar.gz
https://github.com/IKE1997/battery-soh-rul-benchmark/releases/download/v0.1.0/battery_judge.tar.gz
https://github.com/IKE1997/battery-soh-rul-benchmark/releases/download/v0.1.0/SHA256SUMS
```

Large CSV files are intentionally excluded from git because the SE-Bench runtime loads them from the release tarballs. The repository contains the task definition, scorer source, generation/audit scripts, documentation, tests, and checksum manifest.
