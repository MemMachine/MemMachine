# utility for interacting with MemMachine
# There are multiple copies of this file, please keep all of them the same
# 1. memmachine-test/benchmark/amemgym/utils
# 2. memmachine-test/benchmark/longmemeval/utils
# 3. memmachine-test/benchmark/mem0_locomo/tests/memmachine
# 4. memmachine-test/benchmark/mem0_locomo/tests/mods/MemMachine/evaluation/locomo/utils
# ruff: noqa: PTH118, SIM108, G004, C901, SIM105, SIM102

import copy
import os
import sys
from urllib.parse import urlparse

import requests

if True:
    # find path to other scripts and modules
    my_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.abspath(os.path.join(my_dir, ".."))
    top_dir = os.path.abspath(os.path.join(test_dir, ".."))
    utils_dir = os.path.join(top_dir, "utils")
    sys.path.insert(1, test_dir)
    sys.path.insert(1, top_dir)
    from memmachine_helper_base import MemmachineHelperBase


class MemmachineHelperRestapiv1(MemmachineHelperBase):
    """MemMachine REST API v1
    Please use factory method to create this object
    Specification is in MemMachine repo:
        cd MemMachine/src/memmachine/server/api.py
    """

    def __init__(self, log=None, url=None):
        super().__init__()
        self.url = url
        if not self.url:
            self.url = "http://localhost:8080"
        urlobj, host, port = self.split_url(self.url)
        self.urlobj = urlobj
        self.host = host
        self.port = port
        self.cookies = {}
        self.origin = f"{self.urlobj.scheme}://{self.host}"
        if self.port:
            self.origin += f":{self.port}"
        self.api_v1 = f"{self.origin}/v1"
        self.metric_url = f"{self.origin}/metrics"
        self.health_url = f"{self.origin}/health"
        self.mem_add_url = f"{self.api_v1}/memories"
        self.mem_add_episodic_url = f"{self.api_v1}/memories/episodic"
        self.mem_add_semantic_url = f"{self.api_v1}/memories/profile"
        self.mem_search_url = f"{self.api_v1}/memories/search"
        self.mem_search_episodic_url = f"{self.api_v1}/memories/episodic/search"
        self.mem_search_semantic_url = f"{self.api_v1}/memories/profile/search"
        self.metrics_before = {}
        self.metrics_after = {}
        self.rest_variation = 1

    def split_url(self, url):
        urlobj = urlparse(url)
        hostport = urlobj.netloc
        fields = hostport.split(":")
        host = fields[0]
        if len(fields) > 1:
            port = fields[1]
        else:
            port = ""
        return (urlobj, host, port)

    def get_headers(self):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/javascript, */*",
            "X-Requested-With": "XMLHttpRequest",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
            "Accept-Encoding": "gzip, deflate, br",
        }
        headers = copy.copy(headers)
        self.get_origin_referer(headers)
        return headers

    def get_origin_referer(self, headers=None):
        origin = f"{self.urlobj.scheme}://{self.host}"
        if self.port:
            origin += f":{self.port}"
        if headers:
            headers["Origin"] = origin
            headers["Referer"] = f"{origin}/"
        return origin

    #################################################################
    # rest api functions
    #################################################################
    def get_health(self, headers=None, timeout=None):
        """Get memmachine health"""
        return {}
        if not timeout:
            timeout = 30
        def_headers = self.get_headers()
        if headers:
            def_headers.update(headers)
        headers = def_headers
        url = self.health_url
        self.log.debug(f"GET url={url}")
        resp = requests.get(url, headers=headers, cookies=self.cookies, timeout=timeout)
        self.cookies = resp.cookies
        self.log.debug(f"status={resp.status_code} reason={resp.reason}")
        self.log.debug(f"text={resp.text}")
        if resp.status_code < 200 or resp.status_code > 299:
            raise AssertionError(
                f"ERROR: status_code={resp.status_code} reason={resp.reason}"
            )

        data = resp.json()
        return data

    def get_metrics(self, headers=None, timeout=None, quiet=False):
        """Get memmachine metrics"""
        if not timeout:
            timeout = 30
        def_headers = self.get_headers()
        if headers:
            def_headers.update(headers)
        headers = def_headers
        url = self.metric_url
        if not quiet:
            self.log.debug(f"GET url={url}")
        resp = requests.get(url, headers=headers, cookies=self.cookies, timeout=timeout)
        self.cookies = resp.cookies
        if not quiet:
            self.log.debug(f"status={resp.status_code} reason={resp.reason}")
            self.log.debug(f"text={resp.text}")
        if resp.status_code < 200 or resp.status_code > 299:
            raise AssertionError(
                f"ERROR: status_code={resp.status_code} reason={resp.reason}"
            )

        metrics = {}
        for line in resp.text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(" ", 1)
            if len(fields) != 2:
                self.log.error(f"ERROR: cannot split into k,v line={line}")
            else:
                k = fields[0].strip()
                v = fields[1].strip()
                f = None
                i = None
                n = None
                try:
                    f = float(v)
                except Exception:
                    pass
                try:
                    i = int(v)
                except Exception:
                    pass
                if f is not None:
                    if i is None or f > float(i):
                        n = f
                if i is not None and n is None:
                    n = i
                if n is not None:
                    v = n
                metrics[k] = v

        keys = list(metrics.keys())
        for key in keys:
            v = metrics[key]
            if not key.endswith("_created") and not quiet:
                self.log.debug(f"key={key} value={v} type={type(v)}")
        if not self.metrics_before:
            self.metrics_before = metrics
        else:
            self.metrics_after = metrics
        return metrics

    def diff_metrics(self, metrics_before=None, metrics_after=None):
        """Return the diff between two sets of metrics

        Before and after metrics come from get_metrics()
        If before or after is not given, it is taken from internally saved data.
        """
        if not metrics_before:
            metrics_before = self.metrics_before
        if not metrics_after:
            metrics_after = self.metrics_after
        metrics = {}
        if not metrics_before:
            raise AssertionError("ERROR: before metrics not found")
        if not metrics_after:
            raise AssertionError("ERROR: after metrics not found")
        for key in metrics_after:
            diff = None
            before = None
            after = metrics_after[key]
            if key in metrics_before:
                before = metrics_before[key]
            if before is None:
                diff = after
            else:
                is_number = False
                if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                    is_number = True
                if is_number:
                    diff = after - before
                elif before == after:
                    diff = after
                else:
                    diff = f'before="{before}" after="{after}"'
            metrics[key] = diff
        return metrics

    def list_memory(
        self,
        org_id=None,
        project_id=None,
        filter_str=None,
        mem_type=None,
        page_size=None,
        page_num=None,
        headers=None,
        timeout=None,
    ):
        """list memory from memmachine
        See specs in memmachine source code
        Note: if mem_type is None, it defaults to episodic
        Inputs:
            filter_str (str): TBD check memmachine source code
            mem_type (str): <None | episodic | semantic>
        """
        return {}

    def add_memory(
        self,
        org_id=None,
        project_id=None,
        producer=None,
        produced_for=None,
        timestamp=None,
        role=None,
        content=None,
        mem_type=None,
        metadata=None,
        messages=None,
        headers=None,
        timeout=None,
    ):
        """add memory to memmachine
        See specs in memmachine source code
        Note: if mem_type is None, it defaults to both
        Note: must provide either messages or <content, producer>
        Note: every message must have a producer
        Note: must give at least 5 messages to add semantic memory,
              otherwise no semantic memory will be added
        Input:
            mem_type (str): <None | episodic | semantic>
            messages (list of dict): pre-formatted messages to ingest
            if no messages, then one message is built from content
            other params: are added to each message if not present
        """
        if not timeout:
            timeout = 30
        if not messages:
            if not content:
                raise AssertionError("ERROR: no content found")
            messages = [{"content": content}]
        if mem_type:
            mem_type = mem_type.lower()  # memmachine only accepts lower
            if mem_type not in ["none", "episodic", "semantic"]:
                raise AssertionError(f"ERROR: unknown mem_type={mem_type}")
            if mem_type == "none":
                mem_type = None
        if not metadata:
            metadata = {}

        data = []
        for message in messages:
            if "content" not in message or not message["content"]:
                raise AssertionError("ERROR: some messages missing content")
            if "producer" not in message or not message["producer"]:
                raise AssertionError("ERROR: some messages missing producer")

            am_metadata = metadata
            if message["metadata"]:
                am_metadata = message["metadata"]
            am_payload = {
                "session": {
                    "group_id": "test_group",
                    "agent_id": ["test_agent"],
                    "user_id": ["test_user"],
                    "session_id": "session_123",
                },
                "producer": "test_user",
                "produced_for": "test_agent",
                "episode_content": message["content"],
                "episode_type": "message",
                "metadata": am_metadata,
            }
            def_headers = self.get_headers()
            if headers:
                def_headers.update(headers)
            headers = def_headers
            if mem_type and mem_type == "episodic":
                url = self.mem_add_episodic_url
            elif mem_type and mem_type == "semantic":
                url = self.mem_add_semantic_url
            else:
                url = self.mem_add_url
            tmp_payload = copy.copy(am_payload)
            self.log.debug(f"POST url={url} payload={tmp_payload}")
            resp = requests.post(
                url,
                headers=headers,
                json=am_payload,
                cookies=self.cookies,
                timeout=timeout,
            )
            self.cookies = resp.cookies
            self.log.debug(f"status={resp.status_code} reason={resp.reason}")
            self.log.debug(f"text={resp.text}")
            if resp.status_code < 200 or resp.status_code > 299:
                raise AssertionError(
                    f"ERROR: status_code={resp.status_code} reason={resp.reason}"
                )
            try:
                j = resp.json()
            except Exception:
                j = resp.text
            data.append(j)
        return data

    def search_memory(
        self,
        query,
        org_id=None,
        project_id=None,
        top_k=None,
        filter_str=None,
        types=None,
        headers=None,
        timeout=None,
    ):
        """search memory in memmachine
        See specs in memmachine source code
        Note: if types is None, it defaults to all ['episodic', 'semantic']
        Inputs:
            filter_str (str): TBD check memmachine source code
            types (list of str): [<None | episodic | semantic>, ...]
        """
        if not timeout:
            timeout = 30
        if not top_k:
            top_k = 50
        mem_type = None
        if types:
            if len(types) != 1:
                raise AssertionError(
                    f"ERROR: v1 supports only single type types={types}"
                )
            mem_type = types[0]
        sm_payload = {
            "session": {
                "group_id": "test_group",
                "agent_id": ["test_agent"],
                "user_id": ["test_user"],
                "session_id": "session_123",
            },
            "query": query,
            "filter": {},
            "limit": top_k,
        }
        def_headers = self.get_headers()
        if headers:
            def_headers.update(headers)
        headers = def_headers
        if mem_type and mem_type == "episodic":
            url = self.mem_search_episodic_url
        elif mem_type and mem_type == "semantic":
            url = self.mem_search_semantic_url
        else:
            url = self.mem_search_url
        self.log.debug(f"POST url={url} payload={sm_payload}")
        resp = requests.post(
            url, headers=headers, json=sm_payload, cookies=self.cookies, timeout=timeout
        )
        self.cookies = resp.cookies
        self.log.debug(f"status={resp.status_code} reason={resp.reason}")
        self.log.debug(f"text={resp.text}")
        if resp.status_code < 200 or resp.status_code > 299:
            raise AssertionError(
                f"ERROR: status_code={resp.status_code} reason={resp.reason}"
            )

        data = resp.json()
        return data

    #################################################################
    # rest api parsers
    #################################################################
    def build_ltm_ctx(self, data, use_xml=None, do_json_str=None):
        if use_xml is None:
            use_xml = True
        if "content" not in data or "episodic_memory" not in data["content"]:
            return ""
        em = data["content"]["episodic_memory"]
        ltm_ctx = ""
        if len(em) < 1:
            return ""
        ltm_episodes = em[0]
        for episode in ltm_episodes:
            metadata = episode.get("user_metadata", {})
            if "source_timestamp" in metadata:
                ts = metadata["source_timestamp"]
            else:
                ts = episode.get("timestamp", "")
            ts = self.locomo_timestamp_format(ts)
            if "source_speaker" in metadata:
                user = metadata["source_speaker"]
            else:
                user = episode.get("producer_id", "")
            content = episode.get("content")
            if not content:
                continue
            if do_json_str:
                content = self.quote_by_json_str(content)
            ctx = f"[{ts}] {user}: {content}"
            ltm_ctx += f"{ctx}\n"
        if ltm_ctx and use_xml:
            ltm_ctx = (
                f"<LONG TERM MEMORY EPISODES>\n{ltm_ctx}\n</LONG TERM MEMORY EPISODES>"
            )
        return ltm_ctx

    def build_stm_ctx(self, data, use_xml=None, do_json_str=None):
        if use_xml is None:
            use_xml = True
        if "content" not in data or "episodic_memory" not in data["content"]:
            return ""
        em = data["content"]["episodic_memory"]
        stm_ctx = ""
        if len(em) < 2:
            return ""
        stm_episodes = em[1]
        for episode in stm_episodes:
            metadata = episode.get("user_metadata", {})
            if "source_timestamp" in metadata:
                ts = metadata["source_timestamp"]
            else:
                ts = episode.get("timestamp", "")
            ts = self.locomo_timestamp_format(ts)
            if "source_speaker" in metadata:
                user = metadata["source_speaker"]
            else:
                user = episode.get("producer_id", "")
            content = episode.get("content")
            if not content:
                continue
            if do_json_str:
                content = self.quote_by_json_str(content)
            ctx = f"[{ts}] {user}: {content}"
            stm_ctx += f"{ctx}\n"
        if stm_ctx and use_xml:
            stm_ctx = (
                f"<WORKING MEMORY EPISODES>\n{stm_ctx}\n</WORKING MEMORY EPISODES>"
            )
        return stm_ctx

    def build_stm_sum_ctx(self, data, use_xml=None, do_json_str=None):
        if use_xml is None:
            use_xml = True
        if "content" not in data or "episodic_memory" not in data["content"]:
            return ""
        em = data["content"]["episodic_memory"]
        stm_sum_ctx = ""
        if len(em) < 3:
            return ""
        stm_summaries = em[2]
        for stm_summary in stm_summaries:
            if not stm_summary:
                continue
            if do_json_str:
                stm_summary = self.quote_by_json_str(stm_summary)
            stm_sum_ctx += f"{stm_summary}\n"
        if stm_sum_ctx and use_xml:
            stm_sum_ctx = (
                f"<WORKING MEMORY SUMMARY>\n{stm_sum_ctx}\n</WORKING MEMORY SUMMARY>"
            )
        return stm_sum_ctx

    def build_sm_ctx(self, data, use_xml=None, do_json_str=None):
        if use_xml is None:
            use_xml = True
        if "content" not in data or "profile_memory" not in data["content"]:
            return ""
        sm_list = data["content"]["profile_memory"]
        sm_ctx = ""
        print(f"ERROR: @@@@@ not implemented yet sm_list={sm_list}")
        return sm_ctx

    def split_data(self, data):
        """split data returned by search memory into its components

        Inputs:
            data (dict): data returned by search memories
        Outputs:
            le = long term memory episodes
            se = short term memory episodes
            ss = short term memory summaries
            sm = semantic memory facts
        """
        le = []  # long term memory episodes
        se = []  # short term memory episodes
        ss = []  # short term memory summaries
        sm = []  # semantic memory facts
        try:
            content = data.get("content", {})
            em = content.get("episodic_memory", [])
            sm = content.get("profile_memory", [])
            if len(em) > 0:
                le = em[0]
            if len(em) > 1:
                se = em[1]
            if len(em) > 2:
                ss = em[2]
            if ss == [""]:
                ss = []
        except Exception:
            pass
        return le, se, ss, sm
