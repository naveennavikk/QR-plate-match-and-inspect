# QR Plate API Load Test Report

## Summary

The local service remained stable during a small synthetic load test, but this setup is below lakh-scale daily pair capacity. The matcher/inspector was exercised on generated QR plate and property images; no real property images were available in the project workspaces.

## Test environment

- Local Windows host: 4 logical CPUs, 15.9 GB RAM, no CUDA GPU.
- FastAPI service: one Uvicorn process, bounded inference pool of 4 workers, up to 100 pending pairs.
- The load generator ran on the same host as the API.
- Each generated pair used a 720 x 960 reference image and a 1600 x 1200 property image. The JSON request was about 0.51 MB per pair.
- The same synthetic pair was reused for repetitions. It consistently produced automatic accept, so this measures throughput/stability, not matching accuracy.

## Measurements

### 100 single-pair requests, four simultaneous clients

- Wall time: 249.41 seconds
- Throughput: 0.401 pairs/second
- Latency: p50 9.76 seconds, p95 11.27 seconds, maximum 11.40 seconds
- HTTP 200: 100/100
- Successful inference: 100/100
- Automatic accept: 100/100
- Request or inference errors: 0

### One 20-pair batch

- Wall time: 42.46 seconds with default PyTorch CPU thread settings
- A repeat with one PyTorch thread per worker completed in 41.25 seconds
- Inference results: 20/20 successful, all automatically accepted
- Batch body size: about 10.28 MB

The thread-setting repeat was only one sample and the difference was small; it does not establish a reliable speedup.

### Real-image batch and replay

The user-provided file contained 9 distinct image pairs and was valid JSON. The first request, containing those 9 pairs (about 7.71 MB), completed in 110.60 seconds: 9/9 processed, with 1 accept, 2 rejects, and 6 manual reviews.

To exercise the configured 20-pair batch limit, those 9 image pairs were replayed (cycled) to make 20 items. This request was about 17.17 MB and completed in 237.86 seconds: 20/20 processed, with 2 accepts, 5 rejects, and 13 manual reviews. No item errors occurred. The API health check remained `ok` afterward.

This replay rate was 0.084 pairs/second, or about 7,265 pairs/day if sustained continuously. The image pairs were real, but 11 of the 20 entries repeated earlier pairs; this remains a small workload sample rather than a production capacity guarantee.

## Rough continuous capacity estimate

Synthetic throughput corresponded to approximately 34,600–41,900 **pairs** per 24 hours. The real-image replay was substantially slower at a projected 7,265 pairs/day. Since each pair contains two images, that is about 14,530 individual images/day at the replay rate. These are continuous-run projections, not safe production capacity commitments. Real image dimensions, QR scale, localization fallback, traffic peaks, downtime, and external network latency can change them.

Using the real-image replay as the conservative local baseline, 100,000 pairs/day requires about 1.16 pairs/second, roughly 13.8 times the measured rate before adding operational headroom. If “100,000 images” means 50,000 pairs, the average required rate is 0.58 pairs/second, roughly 6.9 times the replay rate before headroom.

## Changes and limits

The batch endpoint now dispatches image pairs through a shared, bounded four-worker inference pool rather than processing one batch serially. A pending-pair cap prevents unlimited in-memory work; when it is full, requests/items receive a capacity error and should be retried. This is not a durable job queue. The endpoint remains synchronous, so large batches wait for all results before returning.

The service is configured for CPU inference on this host. The four-worker configuration is local-test tuning, not a production sizing recommendation. Increasing workers on this four-CPU host can reduce throughput through CPU contention.

## Next capacity tests

1. Run the same load profile with a representative, labeled real-image set to measure accuracy and real latency.
2. Establish average and peak pair arrival rates from the daily workload.
3. Run smoke/baseline, ramp/stress, spike, and soak tests while monitoring p50/p95/p99 latency, throughput, errors, CPU/RAM/GPU, and queue depth. These profiles test different failure modes; see [k6 API load-testing guidance](https://grafana.com/docs/k6/latest/testing-guides/api-load-testing/).
4. Repeat from a separate load-generator machine so its CPU does not compete with inference.
5. For lakh-scale daily processing, deploy a durable queue and multiple inference workers/instances, then rerun the same profile. FastAPI supports multiple worker processes, but each process has its own model memory footprint; size them against available RAM/GPU ([FastAPI worker deployment guidance](https://fastapi.tiangolo.com/deployment/server-workers/)).
