# Results Sharing

Generated research outputs are not committed to the source repository by
default. This avoids mixing code with large models, repeated logs and local
rendering artifacts.

A separately shared result package should contain:

1. the exact versioned configuration;
2. episode-level evaluation data, not only plotted summaries;
3. derived tables used by each figure;
4. a concise interpretation and limitation statement;
5. representative figures or videos;
6. a SHA-256 manifest and the source commit SHA.

Do not upload interrupted runs, private machine paths, credentials or raw files
whose provenance cannot be reconstructed.
