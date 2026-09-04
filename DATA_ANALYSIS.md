# Data Size & Time Analysis for 02_behavioral

## Model: gemma-3-4b-it
- Hidden dimension: 3072
- Total layers: 24
- Dtype: bfloat16 (4 bytes per float)

## Dataset
- MMLU: ~5000 items (primary)
- TriviaQA: ~5000 items (secondary, 5.4 transfer check)
- Conditions: 5 fixed conditions
- **Total generations: 5000 items × 5 conditions = 25,000**

## Storage per layer

**Per item activation:**
- Shape: (3072,) bfloat16
- Size: 3072 × 4 bytes = **12.3 KB**

**Per layer total (25,000 items):**
- 25,000 × 12.3 KB = **307.5 MB per layer**

## Two scenarios

### Scenario A: Initial run (last 12 layers, every other)
Layers: 12, 14, 16, 18, 20, 22 (6 layers)

**Storage:**
- 6 layers × 307.5 MB = **1.84 GB**
- Metadata (item_ids, answers, correct, confidence): ~2.5 MB (negligible)
- **Total: ~1.85 GB**

**Disk I/O (estimate 100 MB/s locally):**
- Write (save): ~18.5 sec
- Read (reload): ~18.5 sec

### Scenario B: Full run (all 24 layers)
**Storage:**
- 24 layers × 307.5 MB = **7.38 GB**
- Metadata: ~2.5 MB (negligible)
- **Total: ~7.40 GB**

**Disk I/O (estimate 100 MB/s locally):**
- Write (save): ~74 sec
- Read (reload): ~74 sec

## Time impact (GPU forward pass excluded)

- **Scenario A overhead (saving/loading):** ~37 sec total
- **Scenario B overhead (saving/loading):** ~148 sec total
- **Difference:** +111 sec for 4× more data

## Recommendation

Start with **Scenario A (6 layers)**. Rationale:
1. 1.85 GB fits comfortably in VRAM scratch + local disk
2. Fast iteration cycle (quick saves/loads)
3. Covers mechanism check (top layers where verbalization lives)
4. If initial probes show clear signal, can expand to all 24 layers
5. Can always re-run with full sweep if needed (costs ~$1–2 extra GPU time)

For full run, ~7.4 GB is still reasonable—only ~2.5 min of disk I/O per full cycle.
