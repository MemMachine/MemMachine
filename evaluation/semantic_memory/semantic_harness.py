import contextlib
import copy

from testcontainers.postgres import PostgresContainer


def apply_cluster_settings(conf: dict, *, similarity_threshold: float) -> None:
    semantic_conf = conf.setdefault("semantic_memory", {})
    semantic_conf["cluster_similarity_threshold"] = similarity_threshold


def build_run_config(base: dict, *, similarity_threshold: float) -> dict:
    run_conf = copy.deepcopy(base)
    apply_cluster_settings(run_conf, similarity_threshold=similarity_threshold)
    return run_conf


@contextlib.contextmanager
def maybe_start_pg_container(enabled: bool):
    if not enabled:
        yield None
        return
    with PostgresContainer("pgvector/pgvector:pg16") as container:
        yield {
            "host": container.get_container_host_ip(),
            "port": int(container.get_exposed_port(5432)),
            "user": container.username,
            "password": container.password,
            "db_name": container.dbname,
        }
