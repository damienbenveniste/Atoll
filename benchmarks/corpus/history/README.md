# Reviewed corpus history

This directory contains compact snapshots created only by a maintainer running
`python -m scripts.benchmark_corpus promote` against a complete evidence slice.
Raw samples, logs, wheels, and cloned source do not belong here.

Snapshot filenames encode the reviewed label, corpus tier, and runner platform.
Never compare snapshots across platforms or tiers, and compare case performance
only when its retained `comparison_key` is identical. Performance snapshots must
retain the exact structured experiment identity stored beside every workflow
case result. Promotion rejects missing identities, mixed run IDs or attempts,
head-SHA mismatches, and independently renamed labels.
