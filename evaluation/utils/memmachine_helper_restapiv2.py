# utility for interacting with MemMachine
# There are multiple copies of this file, please keep all of them the same
# 1. memmachine-test/benchmark/amemgym/utils
# 2. memmachine-test/benchmark/longmemeval/utils
# 3. memmachine-test/benchmark/mem0_locomo/tests/memmachine
# 4. memmachine-test/benchmark/mem0_locomo/tests/mods/MemMachine/evaluation/locomo/utils
# ruff: noqa: PTH118, SIM108, G004, C901, SIM105, SIM102, SIM117, TRY400

import copy
import os
import sys
import threading
import time
import traceback
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
        self.proj_add_url = f"{self.api_v2}/projects"
        self.proj_get_url = f"{self.api_v2}/projects/get"
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

    #################################################################
    # rest api functions
    #################################################################
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

    def add_project(
        self,
        org_id,
        project_id,
        description=None,
        config=None,
        headers=None,
        timeout=None,
    ):
        """add a project
        See specs in memmachine source code

        Note: If config is given, it must have both fields:
            project_config = {
                'reranker': 'mandatory',
                'embedder': 'mandatory',
            }
        """
        if not timeout:
            timeout = 30
        ap_payload = {
            "org_id": org_id,
            "project_id": project_id,
        }
        if description:
            ap_payload["description"] = description
        if config:
            ap_payload["config"] = config
        def_headers = self.get_headers()
        if headers:
            def_headers.update(headers)
        headers = def_headers
        url = self.proj_add_url
        self.log.debug(f"POST url={url} payload={ap_payload}")
        resp = requests.post(
            url,
            headers=headers,
            json=ap_payload,
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
        data = resp.json()
        return data

    def get_project(
        self,
        org_id,
        project_id,
        headers=None,
        timeout=None,
    ):
        """get a project
        See specs in memmachine source code
        """
        if not timeout:
            timeout = 30
        gp_payload = {
            "org_id": org_id,
            "project_id": project_id,
        }
        def_headers = self.get_headers()
        if headers:
            def_headers.update(headers)
        headers = def_headers
        url = self.proj_get_url
        self.log.debug(f"POST url={url} payload={gp_payload}")
        resp = requests.post(
            url,
            headers=headers,
            json=gp_payload,
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
            if self.v2_variation != 2:
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
        if not mem_type or mem_type == "semantic":
            self.semantic_inserting()
        try:
            done = False
            if not done:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=am_payload,
                    cookies=self.cookies,
                    timeout=timeout,
                )
                if resp.status_code >= 200 and resp.status_code <= 299:
                    if self.v2_variation == 0 and mem_type:
                        self.v2_variation = 1  # 3 URLs worked, detected v2_variation 1
                        self.log.debug(
                            f"{mem_type} URL worked, set v2_variation={self.v2_variation}"
                        )
                    done = True
            if not done:
                if self.v2_variation in [0, 1] and url != self.mem_add_url:
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
                        done = True
            if not done:
                self.log.debug("retry v2_variation=2 also failed (1)")
        except Exception as ex:
            if self.v2_variation != 0:
                raise  # real error
            if not mem_type:
                raise  # real error
            if url == self.mem_add_url:
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

    async def async_add_memory(
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
            if self.v2_variation != 2:
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
        if not mem_type or mem_type == "semantic":
            self.semantic_inserting()
        status_code = -1
        reason = ""
        resp_text = ""
        atimeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=atimeout) as session:
            try:
                done = False
                if not done:
                    async with session.post(
                        url,
                        headers=headers,
                        json=am_payload,
                    ) as resp:
                        status_code = resp.status
                        reason = resp.reason
                        resp_text = await resp.text(encoding="utf-8")
                        if resp.status >= 200 and resp.status <= 299:
                            if self.v2_variation == 0 and mem_type:
                                self.v2_variation = (
                                    1  # 3 URLs worked, detected v2_variation 1
                                )
                                self.log.debug(
                                    f"{mem_type} URL worked, set v2_variation={self.v2_variation}"
                                )
                            done = True
                if not done:
                    if self.v2_variation in [0, 1] and url != self.mem_add_url:
                        self.log.debug("v2_variation=1 failed, try v2_variation=2 (1)")
                        url = self.mem_add_url
                        types = [mem_type]
                        am_payload["types"] = types
                        async with session.post(
                            url,
                            headers=headers,
                            json=am_payload,
                        ) as resp:
                            status_code = resp.status
                            reason = resp.reason
                            resp_text = await resp.text(encoding="utf-8")
                            if resp.status >= 200 and resp.status <= 299:
                                self.v2_variation = (
                                    2  # types worked, detected v2_variation 2
                                )
                                self.log.debug(
                                    f"retry types worked, set v2_variation={self.v2_variation} (1)"
                                )
                                done = True
                if not done:
                    self.log.debug("retry v2_variation=2 also failed (1)")
            except Exception as ex:
                if self.v2_variation != 0:
                    raise  # real error
                if not mem_type:
                    raise  # real error
                if url == self.mem_add_url:
                    raise  # real error
                self.log.debug(f"v2_variation=1 failed, try v2_variation=2 (2) ex={ex}")
                url = self.mem_add_url
                types = [mem_type]
                am_payload["types"] = types
                async with session.post(
                    url,
                    headers=headers,
                    json=am_payload,
                ) as resp:
                    status_code = resp.status
                    reason = resp.reason
                    resp_text = await resp.text(encoding="utf-8")
                    if resp.status >= 200 and resp.status <= 299:
                        self.v2_variation = 2  # types worked, detected v2_variation 2
                        self.log.debug(
                            f"retry types worked, set v2_variation={self.v2_variation} (2)"
                        )
                    else:
                        self.log.debug("retry v2_variation=2 also failed (2)")
        self.log.debug(f"status={status_code} reason={reason}")
        self.log.debug(f"text={resp_text}")
        if status_code < 200 or status_code > 299:
            raise AssertionError(f"ERROR: status_code={status_code} reason={reason}")

        data = await resp.json()
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
        ltm = em.get("long_term_memory", {})
        ltm_episodes = ltm.get("episodes", [])
        for episode in ltm_episodes:
            metadata = episode.get("metadata", {})
            if "source_timestamp" in metadata:
                ts = metadata["source_timestamp"]
            else:
                ts = episode.get("created_at", "")
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
        stm = em.get("short_term_memory", {})
        stm_episodes = stm.get("episodes", [])
        for episode in stm_episodes:
            metadata = episode.get("metadata", {})
            if "source_timestamp" in metadata:
                ts = metadata["source_timestamp"]
            else:
                ts = episode.get("created_at", "")
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
        stm = em.get("short_term_memory", {})
        stm_summaries = stm.get("episode_summary", [])
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
        if "content" not in data or "semantic_memory" not in data["content"]:
            return ""
        sm_list = data["content"]["semantic_memory"]
        sm_ctx = ""
        for sm_item in sm_list:
            feature = sm_item.get("feature_name", "")
            if do_json_str:
                value = self.quote_by_json_str(sm_item.get("value", ""))
            else:
                value = sm_item.get("value", "")
            if not feature or not value:
                continue
            sm_ctx += f"- {feature}: {value}\n"
        if sm_ctx and use_xml:
            sm_ctx = f"<SEMANTIC MEMORY>\n{sm_ctx}\n</SEMANTIC MEMORY>"
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
            sm = content.get("semantic_memory", [])
            ltm = em.get("long_term_memory", {})
            stm = em.get("short_term_memory", {})
            le = ltm.get("episodes", [])
            se = stm.get("episodes", [])
            ss = stm.get("episode_summary", [])
            if ss == [""]:
                ss = []
        except Exception:
            pass
        return le, se, ss, sm

    ############################################################
    # rest api semantic memory check functions
    ############################################################
    def start_semantic_check(self, config=None):
        """start semantic memory check thread

        The thread polls metrics periodically to see if semantic memory
        processing is happening in the background.  There is currently
        no API for this.  These functions are not totally accurate but
        it helps with writing benchmarks.

        In order for this workaround to work, the calling program must
        call semantic_is_idle() to check if semantic background processing
        is idle, and wait until it is idle, before calling search_memory().

        Note: this workaround only works with restapiv2 or above.

        Inputs:
            config (dict): optional config
                check_interval - polling interval in secs. def=10 secs.
                idle_after - set is_idle after this many consecutive checks
                insert_cancel_after - cancel insert event after this many consecutive checks

        Outputs:
            thread is started
        """
        if hasattr(self, "semantic_stat") and self.semantic_stat["thread"]:
            self.log.info("semantic check thread already running")
        else:
            self.init_semantic_stat(config=config)
            thread = threading.Thread(
                target=self.semantic_check_thread,
                args=(self.semantic_stat,),
                daemon=True,
            )
            thread.start()
            self.semantic_stat["thread"] = thread

    def init_semantic_stat(self, config=None):
        if not config:
            config = {}
        self.semantic_stat = {
            "count": 0,
            "idle_count": 0,
            "busy_count": 0,
            "last_llm_count": 0,
            "llm_count": 0,
            "is_idle": False,
            "is_inserting": False,
            "insert_count": 0,
            "insert_cancelled": False,
            "check_interval": 0,
            "idle_after": 0,
            "insert_cancel_after": 0,
            "thread": None,
            "quit_requested": False,
        }
        self.semantic_stat["check_interval"] = config.get("check_interval", 10)
        self.semantic_stat["idle_after"] = config.get("idle_after", 3)
        self.semantic_stat["insert_cancel_after"] = config.get(
            "insert_cancel_after", 18
        )

    def semantic_inserting(self):
        """tell this workaround code that add_memory will be called

        In order for this workaround to work, the calling program must
        call semantic_inserting() just before calling add_memory().
        """
        if hasattr(self, "semantic_stat"):
            self.semantic_stat["insert_cancelled"] = False
            self.semantic_stat["is_inserting"] = True
            self.semantic_stat["insert_count"] = 0
            self.log.debug(f"semantic memory is inserting stat={self.semantic_stat}")

    def semantic_is_idle(self):
        """check if semantic background processing is currently idle"""
        is_idle = True
        if hasattr(self, "semantic_stat"):
            # time.sleep(self.semantic_stat['check_interval'])
            is_idle = self.semantic_stat["is_idle"]
            if self.semantic_stat["is_inserting"]:
                is_idle = False
        return is_idle

    def semantic_is_cancelled(self):
        """check if waiting for semantic background processing was cancelled

        I have observed after add_memory, sometimes the background processing
        will not start until 2.5 mins later.  It will run for 2.5 mins.
        We don't have visibility into MemMachine, so we use a counter to cancel
        the waiting after maximum time, so we don't hang forever.
        """
        return self.semantic_stat["insert_cancelled"]

    def semantic_check_thread(self, stat):
        try:
            while not stat["quit_requested"]:
                time.sleep(stat["check_interval"])
                stat["count"] += 1
                try:
                    metrics = self.get_metrics(quiet=True)
                    last_llm_count = stat["last_llm_count"]
                    llm_count = int(
                        metrics["language_model_openai_usage_total_tokens_total"]
                    )
                except Exception as ex:
                    self.log.error(f"get_metrics failed ex={ex}")
                    time.sleep(stat["check_interval"])
                    continue
                stat["llm_count"] = llm_count
                changed = llm_count - last_llm_count
                if changed:
                    # semantic memory is busy
                    stat["idle_count"] = 0
                    stat["busy_count"] += 1
                    stat["is_idle"] = False
                    stat["is_inserting"] = False
                    stat["insert_count"] = 0
                    self.log.debug(f"semantic memory is busy stat={stat}")
                    stat["last_llm_count"] = llm_count
                else:
                    # semantic memory is idle
                    # update idle count
                    stat["idle_count"] += 1
                    if stat["idle_count"] >= stat["idle_after"]:
                        stat["busy_count"] = 0
                        stat["is_idle"] = True
                        if stat["idle_count"] == stat["idle_after"]:
                            if stat["is_inserting"]:
                                self.log.debug(
                                    f"semantic memory is inserting stat={stat}"
                                )
                            else:
                                self.log.debug(f"semantic memory is idle stat={stat}")
                    # update insert count
                    if stat["is_inserting"]:
                        stat["insert_count"] += 1
                        if stat["insert_count"] >= stat["insert_cancel_after"]:
                            stat["insert_cancelled"] = True
                            stat["is_inserting"] = False
                            self.log.error(
                                f"ERROR: semantic memory insert cancelled stat={stat}"
                            )
                            stat["insert_count"] = 0
        except Exception as ex:
            self.log.error(f"ERROR: failed ex={ex}")
            self.log.error(traceback.format_exc())
        finally:
            stat["thread"] = None
            stat["quit_requested"] = False

    ############################################################
