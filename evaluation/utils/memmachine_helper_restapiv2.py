# utility for interacting with MemMachine
# There are 2 copies of this file, please keep them both the same
# 1. memmachine-test/benchmark/mem0_locomo/tests/memmachine
# 2. memmachine-test/benchmark/mem0_locomo/tests/mods/MemMachine/evaluation/locomo/utils
# ruff: noqa: PTH118, SIM108, G004, C901, SIM105, SIM102, ASYNC109, SIM117

import copy
import os
import sys
from urllib.parse import urlparse

import aiohttp
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


class MemmachineHelperRestapiv2(MemmachineHelperBase):
    """MemMachine REST API v2
    Please use factory method to create this object
    Specification is in MemMachine repo:
        cd MemMachine/src/memmachine/server/api_v2
        vi spec.py router.py
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
        self.api_v2 = f"{self.origin}/api/v2"
        self.metric_url = f"{self.api_v2}/metrics"
        self.health_url = f"{self.api_v2}/health"
        self.mem_list_url = f"{self.api_v2}/memories/list"
        self.mem_add_url = f"{self.api_v2}/memories"
        self.mem_add_episodic_url = f"{self.api_v2}/memories/episodic/add"
        self.mem_add_semantic_url = f"{self.api_v2}/memories/semantic/add"
        self.mem_search_url = f"{self.api_v2}/memories/search"
        self.metrics_before = {}
        self.metrics_after = {}
        self.rest_variation = 2
        self.v2_variation = 0

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

    def get_health(self, headers=None, timeout=None):
        """Get memmachine health"""
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

    def get_metrics(self, headers=None, timeout=None):
        """Get memmachine metrics"""
        if not timeout:
            timeout = 30
        def_headers = self.get_headers()
        if headers:
            def_headers.update(headers)
        headers = def_headers
        url = self.metric_url
        self.log.debug(f"GET url={url}")
        resp = requests.get(url, headers=headers, cookies=self.cookies, timeout=timeout)
        self.cookies = resp.cookies
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
            if not key.endswith("_created"):
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
        if not timeout:
            timeout = 30
        lm_payload = {}
        if org_id:
            lm_payload["org_id"] = org_id
        if project_id:
            lm_payload["project_id"] = project_id
        if filter_str:
            lm_payload["filter"] = filter_str
        if mem_type:
            lm_payload["mem_type"] = mem_type
        if page_size:
            lm_payload["page_size"] = page_size
        if page_num:
            lm_payload["org_id"] = page_num
        def_headers = self.get_headers()
        if headers:
            def_headers.update(headers)
        headers = def_headers
        url = self.mem_list_url
        self.log.debug(f"POST url={url} payload={lm_payload}")
        resp = requests.post(
            url, headers=headers, json=lm_payload, cookies=self.cookies, timeout=timeout
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
        types=None,
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
        Note: there are two incompatible versions of restapiv2
            v2_variation = 0:
                undetermined yet
            v2_variation = 1:
                3 URLs for add episodic, semantic, both
            v2_variation = 2:
                1 URL with types parameter
            Initially v2_variation = 0, once we detect 1 or 2, it will be set
            If types is given, then v2_variation will be set to 2
        Input:
            mem_type (str): <None | episodic | semantic>
            messages (list of dict): pre-formatted messages to ingest
            if no messages, then one message is built from content
            other params: are added to each message if not present
            types (list of str): [<None | episodic | semantic>, ...]
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
        if types:
            self.v2_variation = 2
            self.log.debug(f"types given, set v2_variation={self.v2_variation}")
        else:
            if self.v2_variation == 2:
                if mem_type:
                    types = [mem_type]  # convert mem_type to types
                    mem_type = None

        for message in messages:
            if isinstance(message, str):
                message = {"content": message}
            if producer:
                if "producer" not in message or not message["producer"]:
                    message["producer"] = producer
            if produced_for:
                if "produced_for" not in message or not message["produced_for"]:
                    message["produced_for"] = produced_for
            if timestamp:
                if "timestamp" not in message or not message["timestamp"]:
                    message["timestamp"] = timestamp
            if role:
                if "role" not in message or not message["role"]:
                    message["role"] = role
            if metadata:
                if "metadata" not in message or not message["metadata"]:
                    message["metadata"] = metadata
            if "content" not in message or not message["content"]:
                raise AssertionError("ERROR: some messages missing content")
            if "producer" not in message or not message["producer"]:
                raise AssertionError("ERROR: some messages missing producer")

        am_payload = {}
        if org_id:
            am_payload["org_id"] = org_id
        if project_id:
            am_payload["project_id"] = project_id
        if types:
            am_payload["types"] = types
        am_payload["messages"] = messages
        def_headers = self.get_headers()
        if headers:
            def_headers.update(headers)
        headers = def_headers
        if self.v2_variation < 2:
            if mem_type and mem_type == "episodic":
                url = self.mem_add_episodic_url
            elif mem_type and mem_type == "semantic":
                url = self.mem_add_semantic_url
            else:
                url = self.mem_add_url
        else:
            url = self.mem_add_url
        tmp_payload = copy.copy(am_payload)
        tmp_messages = tmp_payload["messages"]
        tmp_msg_len = 0
        for tmp_message in tmp_messages:
            tmp_msg_len += len(tmp_message["content"])
        tmp_payload["messages"] = f"<{tmp_msg_len} bytes>"
        self.log.debug(f"POST url={url} payload={tmp_payload}")
        self.log.debug(f"messages={tmp_messages}")
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=am_payload,
                cookies=self.cookies,
                timeout=timeout,
            )
            if self.v2_variation == 0 and mem_type:
                if resp.status_code >= 200 and resp.status_code <= 299:
                    self.v2_variation = 1  # 3 URLs worked, detected v2_variation 1
                    self.log.debug(
                        f"{mem_type} URL worked, set v2_variation={self.v2_variation}"
                    )
                else:
                    self.log.debug("v2_variation=1 failed, try v2_variation=2 (1)")
                    url = self.mem_add_url
                    types = [mem_type]
                    am_payload["types"] = types
                    resp = requests.post(
                        url,
                        headers=headers,
                        json=am_payload,
                        cookies=self.cookies,
                        timeout=timeout,
                    )
                    if resp.status_code >= 200 and resp.status_code <= 299:
                        self.v2_variation = 2  # types worked, detected v2_variation 2
                        self.log.debug(
                            f"retry types worked, set v2_variation={self.v2_variation} (1)"
                        )
                    else:
                        self.log.debug("retry v2_variation=2 also failed (1)")
        except Exception as ex:
            if self.v2_variation != 0:
                raise  # real error
            if not mem_type:
                raise  # real error
            self.log.debug(f"v2_variation=1 failed, try v2_variation=2 (2) ex={ex}")
            url = self.mem_add_url
            types = [mem_type]
            am_payload["types"] = types
            resp = requests.post(
                url,
                headers=headers,
                json=am_payload,
                cookies=self.cookies,
                timeout=timeout,
            )
            if resp.status_code >= 200 and resp.status_code <= 299:
                self.v2_variation = 2  # types worked, detected v2_variation 2
                self.log.debug(
                    f"retry types worked, set v2_variation={self.v2_variation} (2)"
                )
            else:
                self.log.debug("retry v2_variation=2 also failed (2)")
        self.cookies = resp.cookies
        self.log.debug(f"status={resp.status_code} reason={resp.reason}")
        self.log.debug(f"text={resp.text}")
        if resp.status_code < 200 or resp.status_code > 299:
            raise AssertionError(
                f"ERROR: status_code={resp.status_code} reason={resp.reason}"
            )

        data = resp.json()
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

        sm_payload = {}
        if org_id:
            sm_payload["org_id"] = org_id
        if project_id:
            sm_payload["project_id"] = project_id
        if top_k:
            sm_payload["top_k"] = top_k
        if filter_str:
            sm_payload["filter"] = filter_str
        if types:
            sm_payload["types"] = types
        sm_payload["query"] = query
        def_headers = self.get_headers()
        if headers:
            def_headers.update(headers)
        headers = def_headers
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

    async def async_search_memory(
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

        sm_payload = {}
        if org_id:
            sm_payload["org_id"] = org_id
        if project_id:
            sm_payload["project_id"] = project_id
        if top_k:
            sm_payload["top_k"] = top_k
        if filter_str:
            sm_payload["filter"] = filter_str
        if types:
            sm_payload["types"] = types
        sm_payload["query"] = query
        def_headers = self.get_headers()
        if headers:
            def_headers.update(headers)
        headers = def_headers
        url = self.mem_search_url
        self.log.debug(f"POST url={url} payload={sm_payload}")

        atimeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=atimeout) as session:
            async with session.post(
                url,
                headers=headers,
                json=sm_payload,
            ) as resp:
                self.log.debug(f"status={resp.status} reason={resp.reason}")
                if resp.status < 200 or resp.status > 299:
                    raise AssertionError(
                        f"ERROR: status_code={resp.status} reason={resp.reason}"
                    )
                resp_text = await resp.text(encoding="utf-8")
                self.log.debug(f"text={resp_text}")

        data = await resp.json()
        return data
