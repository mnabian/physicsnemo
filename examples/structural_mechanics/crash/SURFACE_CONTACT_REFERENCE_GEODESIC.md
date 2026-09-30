# Optional gap-scaled reference-edge geodesic exclusion

All variants always exclude a node from its incident facets. The new optional
filter excludes an additional query-node/facet pair when

`min(d0(query, vertex) for vertex in facet) < distance_scale * G(query, facet)`

where `G = max(gap_min, gap_scale * (node_gap + facet_gap))`. In this recipe,
`node_gap` is half the supplied nodal shell thickness and `facet_gap` is half
the maximum supplied thickness among the facet vertices. The default distance
multiplier is sqrt(2), gap multiplier is 1, and gap floor is zero. All quantities
are evaluated in physical coordinates/units (mm for GM). The predictive
message activation band (5 mm) does **not** enter this equation.

`d0` is shortest-path length on the original triangle/quad **perimeter-edge**
graph, weighted by initial physical edge lengths. No quad diagonal, spatial
neighbor edge, current-deformed edge length or future target is inserted.
Disconnected surfaces stay eligible even when coincident. A folded but
materially distant region remains eligible. This is a graph-edge approximation
to reference-surface geodesics; it can overestimate continuous-surface distances
and consequently retain pairs that a continuous-distance filter would exclude.

The method is inspired by Radioss TYPE7 `Irem_gap=2`, remark 14:
https://help.altair.com/hwsolvers/rad/topics/solvers/rad/inter_type7_starter_r.htm
It is **not** exact solver replication: we lack the deck, facet/part contact
sets, element thickness metadata and solver gap options. Maximum facet-vertex
thickness is an explicit nodal-data approximation. Mesh-size gap corrections,
tied-contact exclusions, adaptive erosion and friction are not added here.

Configurations:

- `gm_crash_deformer_geodesic_surface_contact_autoregressive_tbptt`: zero floor,
  thickness-based gap only.
- `gm_crash_deformer_geodesic_gap5_surface_contact_autoregressive_tbptt`: explicit
  5 mm gap floor. This is a separate modeling assumption, not an assertion about
  the dataset's solver. It leaves the independent message activation band alone.
- `datapipe.contact_surface_exclusion=incidence`: omit the optional filter.
- `datapipe.contact_geodesic_distance_scale=0`: return incidence exclusions only.

The datapipe copies initial coordinates **before normalization**, and stores
the resulting frozen node/facet mask. Reuse requires identical topology,
initial geometry and thickness, not topology alone. PyG batches offset node and
facet IDs independently. This discrete preprocessing is intentionally detached;
the live closest-point geometry, interpolation and contact messages keep their
existing BPTT derivatives.

## Runtime cache and distributed startup

`datapipe.contact_geodesic_cache_dir` enables a persistent CPU mask cache; the
geodesic recipes default it below the statistics directory. Use the same explicit
directory across restart phases to avoid recomputation. A content key includes
the physical initial coordinates, complete topology and raw thickness (including
shape/dtype), every geodesic option, cache schema and the functional source hash.
Results are checksummed, shape/range/order checked on load, locked per key and
published atomically. Corrupt entries fail closed. Process-local hits still share
the same tensor. Logs expose compute starts and computed/disk/memory hit counts.
The cache contains only static masks, never learned weights or future states.

Dataset startup uses a separate Gloo CPU group with a one-hour bounded timeout
(`training.data_startup_timeout_seconds`, optional override). Rank zero first
publishes training statistics and warms BOTH train and validation masks. Other
ranks then load these artifacts. All ranks finish data initialization before
model/NCCL communication begins. Preprocessing exceptions propagate to all ranks;
process crashes are bounded by the CPU timeout. The training/NCCL timeout, model,
loss, data split, contact eligibility and 500-epoch LR schedule are unchanged.

Bounded Dijkstra does not allocate a dense all-pairs distance matrix. Exclusions
are canonical, sorted and unique; exceeding the explicit storage safety budget
raises instead of truncating. Tests compare against an independent tiny dense
Floyd-Warshall oracle, check strict thresholds and disconnected/folded meshes,
validate physical-coordinate caching, and check checkpointed BPTT gradients.

On the first inspected GM case (Run100), thickness-only preprocessing returned
exactly the incidence set. Do not claim a meaningful distinct training ablation
unless an observed-geometry audit demonstrates additional eligible-pair removal.
