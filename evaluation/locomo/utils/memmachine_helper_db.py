# utility for interacting with MemMachine
# There are 2 copies of this file, please keep them both the same
# 1. memmachine-test/benchmark/mem0_locomo/tests/memmachine
# 2. memmachine-test/benchmark/mem0_locomo/tests/mods/MemMachine/evaluation/locomo/utils
# ruff: noqa: SIM108, RUF059, C901, UP031, SIM102

import psycopg2
from neo4j import GraphDatabase


class Neo4jHelper:
    """MemMachine Neo4j helper
    Check if we added any episodic memories
    """

    def __init__(
        self, log=None, host=None, port=None, dbname=None, user=None, password=None
    ):
        if log:
            self.printmsg = log.debug
        else:
            self.printmsg = self.my_print
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        if not self.host:
            self.host = "localhost"
        if not self.port:
            self.port = 7687
        if not self.dbname:
            self.dbname = "neo4j"
        if not self.user:
            self.user = "neo4j"
        if not self.password:
            self.password = "Password123"
        self.n4_driver = None
        self.clear_stats()

    def my_print(self, msg):
        print(msg)

    def clear_stats(self):
        self.n4_stat = {
            "nodes_before": 0,
            "nodes_after": 0,
            "rels_before": 0,
            "rels_after": 0,
            "nodes": 0,
            "relationships": 0,
            "info": {},
        }
        self.before_counts_saved = False
        self.after_counts_saved = False

    def connect(self):
        if not self.n4_driver:
            n4_url = f"bolt://{self.host}:{self.port}"
            n4_auth = (self.user, self.password)
            self.n4_driver = GraphDatabase.driver(n4_url, auth=n4_auth)
            self.n4_driver.verify_connectivity()
            n4_info = self.n4_driver.get_server_info()
            self.n4_stat["info"] = f"{n4_info.agent}"
        self.printmsg(f"connect completed host={self.host}")

    def disconnect(self):
        if self.n4_driver:
            n4_driver = self.n4_driver
            self.n4_driver = None
            n4_driver.close()
        self.printmsg("disconnect completed")

    def get_db_stats(self, n4_stat=None, pos=None):
        if not n4_stat:
            n4_stat = self.n4_stat
        if not pos:
            if self.before_counts_saved:
                pos = "after"
            else:
                pos = "before"
        if pos:
            pos = pos.lower()
        if pos not in ["before", "after"]:
            raise AssertionError(f"ERROR: unknown pos={pos}")
        if not self.n4_driver:
            self.connect()
        records, summary, keys = self.n4_driver.execute_query(
            "MATCH (n) RETURN count(n) AS node_count;", database_=self.dbname
        )
        node_count = 0
        for record in records:
            node_count += record["node_count"]

        records, summary, keys = self.n4_driver.execute_query(
            "MATCH ()-[r]->() RETURN count(r) AS relationship_count;",
            database_=self.dbname,
        )
        rel_count = 0
        for record in records:
            rel_count += record["relationship_count"]

        if pos == "before":
            n4_stat["nodes_before"] = node_count
            n4_stat["rels_before"] = rel_count
            self.before_counts_saved = True
        else:
            n4_stat["nodes_after"] = node_count
            n4_stat["rels_after"] = rel_count
            n4_stat["nodes"] = n4_stat["nodes_after"] - n4_stat["nodes_before"]
            n4_stat["relationships"] = n4_stat["rels_after"] - n4_stat["rels_before"]
            self.after_counts_saved = True
        self.printmsg(f"get_db_stats completed pos={pos} n4_stat={n4_stat}")
        return n4_stat


class PsqlHelper:
    """MemMachine PSQL helper
    Check if we added any semantic memories
    """

    def __init__(self, log=None, host=None, dbname=None, user=None, password=None):
        if log:
            self.printmsg = log.debug
        else:
            self.printmsg = self.my_print
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        if not self.host:
            self.host = "localhost"
        if not self.dbname:
            self.dbname = "memmachine"
        if not self.user:
            self.user = "memmachine"
        if not self.password:
            self.password = "Password123"
        self.pg_conn = None
        self.clear_stats()

    def my_print(self, msg):
        print(msg)

    def clear_stats(self):
        self.db_tables = {}
        self.delta_tables = {}
        self.before_counts_saved = False
        self.after_counts_saved = False
        self.expected_tables = [
            "episodestore",
            "feature",
            "citations",
            "set_ingested_history",
        ]

    def connect(self):
        if not self.pg_conn:
            dsn = f"host={self.host} dbname={self.dbname} user={self.user} password={self.password}"
            self.pg_conn = psycopg2.connect(dsn)
        else:
            self.pg_conn.rollback()
        self.printmsg(f"connect completed host={self.host}")

    def disconnect(self):
        if self.pg_conn:
            pg_conn = self.pg_conn
            self.pg_conn = None
            pg_conn.close()
        self.printmsg("disconnect completed")

    def db_get_tables(self, db_tables=None):
        if db_tables is None:
            db_tables = self.db_tables
        if not self.pg_conn:
            self.connect()
        with self.pg_conn.cursor() as my_cur:
            query_sql = """SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"""
            my_cur.execute(query_sql)
            rows = my_cur.fetchall()
            for row in rows:
                table_name = row[0]
                if table_name not in db_tables:
                    db_tables[table_name] = {
                        "count_before": 0,
                        "count_after": 0,
                    }
        if not self.db_tables:
            self.db_tables = db_tables
        self.printmsg(f"db_get_tables completed num_tables={len(rows)}")
        return db_tables

    def table_get_row_counts(self, db_tables=None, pos=None):
        if not db_tables:
            db_tables = self.db_tables
        if not db_tables:
            self.db_get_tables()
        if not pos:
            if self.before_counts_saved:
                pos = "after"
            else:
                pos = "before"
        if pos:
            pos = pos.lower()
        if pos not in ["before", "after"]:
            raise AssertionError(f"ERROR: unknown pos={pos}")
        if not self.pg_conn:
            self.connect()
        with self.pg_conn.cursor() as my_cur:
            for db_table in db_tables:
                query_sql = """select count(*) from %s;""" % db_table
                my_cur.execute(query_sql)
                row = my_cur.fetchone()
                count = row[0]
                if pos == "before":
                    db_tables[db_table]["count_before"] = count
                else:
                    db_tables[db_table]["count_after"] = count
                self.printmsg(f"table={db_table} count={count}")
        if not self.before_counts_saved:
            self.before_counts_saved = True
        else:
            self.after_counts_saved = True
        self.printmsg(f"table_get_row_counts completed pos={pos}")
        return db_tables

    def diff_tables(self, db_tables=None):
        if not db_tables:
            db_tables = self.db_tables
        delta_tables = {}
        for name, data in db_tables.items():
            count = data["count_after"] - data["count_before"]
            delta_tables[name] = {"count": count}
        self.delta_tables = delta_tables
        self.printmsg("delta_tables completed")
        return delta_tables

    def check_table_names(self, db_tables=None):
        if not db_tables:
            db_tables = self.db_tables
        missing = []
        for table_name in self.expected_tables:
            if table_name not in db_tables:
                missing.append(table_name)
        self.printmsg(f"check_table_names completed missing={missing}")
        return missing

    def check_new_semantic(self, db_tables=None):
        if not db_tables:
            db_tables = self.db_tables
        errmsgs = ""
        if not errmsgs:
            if not db_tables:
                errmsgs += "ERROR: tables not found\n"
        if not errmsgs:
            missing = self.check_table_names(db_tables)
            if missing:
                errmsgs += f"ERROR: missing tables {missing}\n"
        if not errmsgs:
            if not self.after_counts_saved:
                self.table_get_row_counts()
        if not errmsgs:
            if "count" not in db_tables["feature"]:
                db_tables = self.diff_tables(db_tables)
        if not errmsgs:
            no_inserts = []
            for table_name in self.expected_tables:
                count = db_tables[table_name]["count"]
                if count == 0:
                    no_inserts.append(table_name)
            if no_inserts:
                errmsgs += f"ERROR: these tables have no new rows {no_inserts}\n"
        return errmsgs


if __name__ == "__main__":
    host = "ai3"
    psql = PsqlHelper(host=host)
    print("get before rows")
    psql.table_get_row_counts()
    print("get after rows")
    psql.table_get_row_counts()
    print("mod the after counts")
    for db_table in psql.db_tables:
        psql.db_tables[db_table]["count_after"] += 123
    print("check after rows")
    errmsgs = psql.check_new_semantic()
    print(f"errmsgs={errmsgs}")
    psql.disconnect()
