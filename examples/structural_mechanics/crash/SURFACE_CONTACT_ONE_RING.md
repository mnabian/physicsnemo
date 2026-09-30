# Material one-ring contact ablation

Use `gm_crash_deformer_one_ring_surface_contact_autoregressive_tbptt` to
compare against the unchanged predictive-surface-contact recipe. Both use
standard FLARE, the same model parameters, loss, four-step TBPTT windows,
5 mm activation band, 5 ms causal forecast and 500-epoch learning-rate schedule.

The sole scientific change is the eligible node/facet set. A node is excluded
from a facet if it shares an **original element** with any vertex of that facet.
This includes incident facets and one surrounding material ring. All vertices
of a quad share material membership, including the diagonal vertices. Padded
triangles count only once. We do not use spatial kNN edges, future positions,
component-wide masks, normals or unavailable solver contact-set definitions.

The sparse CPU preprocessing is `(B @ B.T) @ B` with Boolean node/facet
incidence `B`. It is computed once per distinct dataset topology and reused
when the existing datapipe confirms that topology is identical. PyG batching
offsets node IDs and facet IDs independently. Candidate pages filter against
sorted exclusion codes before narrow-phase geometry, without truncating pairs.
Surviving pairs retain the existing live geometry and BPTT gradients.

This is an explicit modeling assumption, not a reconstruction of LS-DYNA or
OpenRadioss. Distant material regions of the same component remain eligible for
self-contact, and disconnected surfaces remain eligible even when coincident.
However, the one-ring mask can suppress genuine contact in a sharply folded
immediate neighborhood. A smaller graph alone is not proof of greater physical
accuracy; evaluate full-rollout validation and held-out deformation errors.

`last_discovery_stats.candidates` counts candidates **after** incidence and
optional topology exclusions. `active_pairs` counts contact-message pairs
selected by the current/predictive gap band, not solver-confirmed contacts.
Diagnostics should report the pre/post ablation counts on identical geometry
and current nonpositive gaps separately; no precision/recall can be claimed
without solver contact labels.
