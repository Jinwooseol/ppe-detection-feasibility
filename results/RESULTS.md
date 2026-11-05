### Split

|  | held-out photographers (used) | random images (baseline) |
|---|---|---|
| images (train / val / test) | 5,649 / 822 / 1,628 | 5,649 / 822 / 1,628 |
| photographers in test | 307 | 321 |
| largest photographer's share of test | 7.5% | 9.4% |
| test images whose photographer is also in train | 0.0% | 91.9% |
| test images with a near-duplicate in train | 0.0% | 0.2% |

### Training (YOLO26s from COCO weights, 640 px, batch 16, seed 0)

| model | epochs | best epoch (val mAP@0.5:0.95) | train minutes | resumed after a fault |
|---|---|---|---|---|
| held-out photographers | 60 | 40 | 49.3 | no |
| random images | 80 | 63 | 65.3 | no |

### Headline

| test set | images | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| held-out photographers | 1628 | 0.490 | 0.315 |
| random images (leaky) | 1628 | 0.635 | 0.405 |

### Per class (held-out photographers)

| class | test images | instances | mAP@0.5 | mAP@0.5:0.95 | mAP@0.5 random split | threshold @ 20/1k | precision % | recall % |
|---|---|---|---|---|---|---|---|---|
| hands | 1295 | 3281 | 0.833 | 0.569 | 0.852 | 0.78 | 98.1 | 49.5 |
| person | 1510 | 2872 | 0.883 | 0.702 | 0.880 | 0.89 | 97.2 | 38.0 |
| head | 1311 | 2633 | 0.874 | 0.654 | 0.886 | 0.73 | 98.3 | 68.2 |
| face | 1118 | 1889 | 0.878 | 0.660 | 0.901 | 0.71 | 97.8 | 74.0 |
| ear | 954 | 1525 | 0.812 | 0.522 | 0.796 | 0.62 | 96.8 | 64.3 |
| shoes | 317 | 1006 | 0.497 | 0.277 | 0.614 | 0.71 | 87.9 | 23.1 |
| tool | 441 | 704 | 0.102 | 0.058 | 0.412 | 0.60 | 55.6 | 5.7 |
| gloves | 264 | 522 | 0.501 | 0.310 | 0.573 | 0.53 | 84.7 | 33.9 |
| glasses | 320 | 402 | 0.622 | 0.333 | 0.643 | 0.47 | 87.2 | 54.2 |
| helmet | 94 | 182 | 0.594 | 0.356 | 0.571 | 0.27 | 72.2 | 45.6 |
| face-mask | 79 | 149 | 0.390 | 0.208 | 0.699 | 0.28 | 62.4 | 35.6 |
| foot | 70 | 146 | 0.186 | 0.087 | 0.266 | 0.19 | 43.9 | 17.1 |
| safety-vest | 43 | 112 | 0.263 | 0.129 | 0.294 | 0.33 | 50.0 | 28.6 |
| ear-mufs | 45 | 86 | 0.184 | 0.078 | 0.504 | 0.22 | 36.0 | 20.9 |
| safety-suit | 28 | 51 | 0.130 | 0.090 | 0.448 | 0.15 | 17.9 | 13.7 |
| face-guard | 25 | 26 | 0.170 | 0.073 | 0.638 | 0.13 | 17.9 | 26.9 |
| medical-suit | 16 | 25 | 0.407 | 0.249 | 0.821 | 0.01 | 28.9 | 52.0 |

### Look-alike confusions at the operating point

| true class | predicted as | boxes | true boxes in test | share of true boxes |
|---|---|---|---|---|
| helmet | head | 1 | 182 | 0.5% |
| head | helmet | 19 | 2633 | 0.7% |
| gloves | hands | 10 | 522 | 1.9% |
| hands | gloves | 14 | 3281 | 0.4% |
| shoes | foot | 13 | 1006 | 1.3% |
| foot | shoes | 16 | 146 | 11.0% |
| face-mask | face | 2 | 149 | 1.3% |
| face | face-mask | 4 | 1889 | 0.2% |
| face | face-guard | 1 | 1889 | 0.1% |
| safety-vest | safety-suit | 1 | 112 | 0.9% |
| safety-suit | safety-vest | 1 | 51 | 2.0% |
| medical-suit | safety-suit | 1 | 25 | 4.0% |

### Manual audit of the look-alike cases (one reviewer, sheets in `results/audit/`)

| label → predicted | cases | item visible, not labelled | edge of the label rule | model wrong | unclear | not audited |
|---|---|---|---|---|---|---|
| helmet → head | 1 | 0 | 0 | 0 | 1 | 0 |
| head → helmet | 19 | 12 | 0 | 5 | 2 | 0 |
| gloves → hands | 10 | 0 | 0 | 10 | 0 | 0 |
| hands → gloves | 14 | 3 | 0 | 4 | 7 | 0 |
| shoes → foot | 13 | 0 | 9 | 3 | 1 | 0 |
| foot → shoes | 16 | 0 | 15 | 0 | 1 | 0 |

### Where the misses come from

| class | true boxes | found % | called look-alike % | poor box % | not detected % | recall % by size: tiny / small / medium / large |
|---|---|---|---|---|---|---|
| helmet | 182 | 45.6 | 1.6 | 0.5 | 52.2 | 16.9 / 35.1 / 77.2 / 82.3 |
| head | 2633 | 68.2 | 1.2 | 0.5 | 30.0 | 2.6 / 26.6 / 85.0 / 88.0 |
| gloves | 522 | 33.9 | 1.9 | 3.1 | 61.1 | 2.1 / 26.3 / 45.1 / 44.6 |
| hands | 3281 | 49.5 | 0.5 | 2.6 | 47.3 | 0.3 / 15.3 / 62.8 / 70.9 |
| shoes | 1006 | 23.1 | 1.4 | 1.7 | 73.9 | 0.3 / 17.0 / 46.7 / 23.0 |
| foot | 146 | 17.1 | 11.0 | 3.4 | 68.5 | 0.0 / 21.4 / 22.9 / 0.0 |
| safety-vest | 112 | 28.6 | 0.9 | 1.8 | 68.8 | 0.0 / 22.5 / 41.9 / 52.6 |
| person | 2872 | 38.0 | 1.3 | 5.0 | 55.6 | 0.0 / 0.0 / 0.0 / 47.2 |

### Operating points (threshold · precision · recall)

| class | ≤ 5 false alerts / 1k frames | ≤ 20 false alerts / 1k frames | ≤ 50 false alerts / 1k frames | default conf 0.25 |
|---|---|---|---|---|
| helmet | 0.63 · P 86.0 · R 26.9 | 0.27 · P 72.2 · R 45.6 | 0.11 · P 58.9 · R 63.7 | P 70.7 · R 47.8 · 22/1k |
| head | 0.86 · P 99.3 · R 42.2 | 0.73 · P 98.3 · R 68.2 | 0.50 · P 96.2 · R 78.2 | P 92.1 · R 84.1 · 117/1k |
| gloves | 0.72 · P 93.5 · R 22.2 | 0.53 · P 84.7 · R 33.9 | 0.34 · P 73.0 · R 42.0 | P 64.8 · R 46.2 · 80/1k |
| hands | 0.86 · P 98.8 · R 20.2 | 0.78 · P 98.1 · R 49.5 | 0.69 · P 96.2 · R 62.3 | P 82.3 · R 79.6 · 345/1k |
| shoes | 0.81 · P 91.7 · R 8.7 | 0.71 · P 87.9 · R 23.1 | 0.52 · P 82.0 · R 36.6 | P 64.1 · R 48.1 · 166/1k |
| foot | 0.34 · P 63.6 · R 9.6 | 0.19 · P 43.9 · R 17.1 | 0.12 · P 32.5 · R 26.7 | P 53.8 · R 14.4 · 11/1k |
| safety-vest | 0.74 · P 55.6 · R 8.9 | 0.33 · P 50.0 · R 28.6 | 0.13 · P 34.7 · R 38.4 | P 42.5 · R 30.4 · 28/1k |
| person | 0.93 · P 98.0 · R 13.5 | 0.89 · P 97.2 · R 38.0 | 0.75 · P 96.0 · R 68.1 | P 84.1 · R 86.4 · 289/1k |

### Deployment (NVIDIA GeForce RTX 5070 Ti Laptop GPU, TensorRT 10.16.1.11, 640 px)

| format | mAP@0.5 | mAP@0.5:0.95 | model latency ms (b1) | model images/s (b1) | model images/s (b8) | per-frame ms (b1) | FPS (b1) | FPS incl. JPEG decode |
|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.490 | 0.315 | 6.40 | 156 | – | unstable 59.9–228 | – | – |
| PyTorch FP16 | 0.489 | 0.315 | 6.72 | 149 | 638 | unstable 3.8–6 | – | – |
| TensorRT FP32 | 0.489 | 0.314 | 2.11 | 474 | – | 3.8 | 261 | 100 |
| TensorRT FP16 | 0.489 | 0.314 | 1.00 | 996 | 1426 | 2.9 | 341 | 109 |

Not built in this run: TensorRT INT8 (RuntimeError: RuntimeError: TensorRT engine build failed, check logs for errors)
