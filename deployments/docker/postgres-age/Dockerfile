# Apache AGE + pgvector in a single PostgreSQL image.
#
# The upstream ``apache/age`` images only bundle the AGE extension. MemMachine's
# ``vector_graph_store`` with ``provider: age`` also needs pgvector for the
# similarity side tables, so this Dockerfile apt-installs
# ``postgresql-16-pgvector`` on top of the official AGE image. The base is
# pinned to a specific release tag to keep builds reproducible.
ARG AGE_BASE=apache/age:release_PG16_1.6.0
FROM ${AGE_BASE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-16-pgvector \
    && rm -rf /var/lib/apt/lists/*
