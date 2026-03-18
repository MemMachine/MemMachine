# utility for interacting with MemMachine
# There are multiple copies of this file, please keep all of them the same
# 1. memmachine-test/benchmark/amemgym/utils
# 2. memmachine-test/benchmark/longmemeval/utils
# 3. memmachine-test/benchmark/mem0_locomo/tests/memmachine
# 4. memmachine-test/benchmark/mem0_locomo/tests/mods/MemMachine/evaluation/locomo/utils
# ruff: noqa: PTH118, C901, RUF059, SIM108

import json
import logging
import os
import sys
from datetime import datetime

if True:
    # find path to other scripts and modules
    my_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.abspath(os.path.join(my_dir, ".."))
    top_dir = os.path.abspath(os.path.join(test_dir, ".."))
    utils_dir = os.path.join(top_dir, "utils")
    sys.path.insert(1, test_dir)
    sys.path.insert(1, top_dir)
    from utils.atf_helper import get_logger


class MemmachineHelperBase:
    """MemMachine helper
    Utility functions to help use MemMachine memory data
    Make MemMachine calls using various interfaces
    """

    def __init__(self, log=None, log_dir=None, debug=None):
        self.log = log
        self.log_dir = log_dir
        self.debug = debug
        if not self.log_dir:
            self.log_dir = "."
        if not log:
            self.log = get_logger(
                log_file=f"{self.log_dir}/memmachine_helper.log",  # atf expects full path here
                log_name="memmachine_helper",
                log_console=True,
            )
            # self.log.setLevel(logging.DEBUG)
            # logging.basicConfig(level=logging.DEBUG)
            log_level = logging.INFO
            if self.debug:
                log_level = logging.DEBUG
            for handler in self.log.handlers:
                if hasattr(handler, "baseFilename"):
                    handler.setLevel(logging.DEBUG)
                else:
                    handler.setLevel(log_level)
        self.rest_variation = 0

    def check_rest_variation(self, data):
        if self.rest_variation > 0:
            return
        if "content" not in data:
            raise AssertionError(f"ERROR: missing content data={data}")
        if "episodic_memory" in data["content"]:
            em = data["content"]["episodic_memory"]
            if isinstance(em, list):
                self.rest_variation = 1
            elif isinstance(em, dict):
                self.rest_variation = 2
        elif "profile_memory" in data["content"]:
            self.rest_variation = 1
        elif "semantic_memory" in data["content"]:
            self.rest_variation = 2
        if not self.rest_variation:
            raise AssertionError(f"ERROR: cannot parse variation from data={data}")

    def build_episodic_ctx(
        self,
        data,
        use_xml=None,
        do_short_term=None,
        do_summary=None,
        do_json_str=None,
    ):
        """combine data returned by search memory into a context text string

        Only episodic memory is used.  Semantic memory is not added into context.

        Inputs:
            data (dict): data returned by search memories
            use_xml (tri-state): choose whether to use xml tags to distinguish memory types
                None: auto, use xml tags if multiple memory types, otherwise do not use xml
                True: always use xml tags
                False: never use xml tags
            do_summary (tri-state): include summary if True or None
        Outputs:
            ctx (str): context string, add question, then feed into LLM
        """
        return self.build_ctx(
            data,
            use_xml=use_xml,
            do_short_term=do_short_term,
            do_summary=do_summary,
            do_json_str=do_json_str,
            do_episodic=True,
            do_semantic=False,
        )

    def build_semantic_ctx(
        self,
        data,
        use_xml=None,
        do_short_term=None,
        do_summary=None,
        do_json_str=None,
    ):
        """combine data returned by search memory into a context text string

        Only semantic memory is used.  Episodic memory is not added into context.

        Inputs:
            data (dict): data returned by search memories
            use_xml (tri-state): choose whether to use xml tags to distinguish memory types
                None: auto, use xml tags if multiple memory types, otherwise do not use xml
                True: always use xml tags
                False: never use xml tags
            do_summary (bool): include summary only if True
        Outputs:
            ctx (str): context string, add question, then feed into LLM
        """
        if do_summary is None:
            do_summary = False
        return self.build_ctx(
            data,
            use_xml=use_xml,
            do_short_term=do_short_term,
            do_summary=do_summary,
            do_json_str=do_json_str,
            do_episodic=False,
            do_semantic=True,
        )

    def build_ctx(
        self,
        data,
        use_xml=None,
        do_short_term=None,
        do_summary=None,
        do_json_str=None,
        do_episodic=None,
        do_semantic=None,
    ):
        """combine data returned by search memory into a context text string

        Both episodic and semanic memory are used.

        Inputs:
            data (dict): data returned by search memories
            use_xml (tri-state): choose whether to use xml tags to distinguish memory types
                None: auto, use xml tags if multiple memory types, otherwise do not use xml
                True: always use xml tags
                False: never use xml tags
            do_summary (tri-state): include summary if True or None
            do_episodic (tri-state): include episodic if True or None
            do_semantic (tri-state): include semantic if True or None
        Outputs:
            ctx (str): context string, add question, then feed into LLM
        """
        if do_short_term is None:
            do_short_term = True
        if do_summary is None:
            do_summary = True
        if do_episodic is None:
            do_episodic = True
        if do_semantic is None:
            do_semantic = True
        num_types, le_len, se_len, ss_len, sm_len = self.split_data_count(data)
        if use_xml is None:  # None is auto
            if num_types > 1:
                use_xml = True  # more than 1 type of memory, use xml to distinguish
            else:
                use_xml = False  # only 1 type of memory, no need to use xml

        ctx = ""
        if not self.rest_variation:
            self.check_rest_variation(data)

        # fmt: off
        ltm_ctx = self.build_ltm_ctx(
            data, use_xml=use_xml, do_json_str=do_json_str
        )
        stm_ctx = self.build_stm_ctx(
            data, use_xml=use_xml, do_json_str=do_json_str
        )
        stm_sum_ctx = self.build_stm_sum_ctx(
            data, use_xml=use_xml, do_json_str=do_json_str
        )
        sm_ctx = self.build_sm_ctx(
            data, use_xml=use_xml, do_json_str=do_json_str
        )
        # fmt: on
        if ltm_ctx and do_episodic:
            ctx += ltm_ctx
            ctx += "\n"
        if stm_ctx and do_episodic and do_short_term:
            ctx += stm_ctx
            ctx += "\n"
        if stm_sum_ctx and do_summary:
            ctx += stm_sum_ctx
            ctx += "\n"
        if sm_ctx and do_semantic:
            ctx += sm_ctx
            ctx += "\n"
        return ctx

    def split_data_count(self, data):
        le, se, ss, sm = self.split_data(data)
        num_types = 0
        le_len = len(le)
        se_len = len(se)
        ss_len = len(ss)
        sm_len = len(sm)
        if le_len:
            num_types += 1
        if se_len:
            num_types += 1
        if ss_len:
            num_types += 1
        if sm_len:
            num_types += 1
        return num_types, le_len, se_len, ss_len, sm_len

    def locomo_timestamp_format(self, timestamp_str):
        """timestamp format that is in locomo10.json dataset
        when running the mem0 evaluation of locomo benchmark.
        We use this to replicate Edwin's benchmark scores
        """
        new_str = timestamp_str
        try:
            ts_obj = datetime.fromisoformat(timestamp_str)
            date_str = ts_obj.date().strftime("%A, %B %d, %Y")
            time_str = ts_obj.time().strftime("%I:%M %p")
            new_str = f"{date_str} at {time_str}"
        except Exception:
            pass
        return new_str

    def quote_by_json_str(self, content):
        try:
            new_content = json.dumps(f"{content}")
        except Exception:
            new_content = content
        return new_content

    #################################################################
    # rest api functions
    #################################################################
    def get_health(self, headers=None, timeout=None):
        raise NotImplementedError

    def get_metrics(self, headers=None, timeout=None, quiet=False):
        raise NotImplementedError

    def diff_metrics(self, metrics_before=None, metrics_after=None):
        raise NotImplementedError

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
        raise NotImplementedError

    def add_project(
        self,
        org_id,
        project_id,
        description=None,
        config=None,
        headers=None,
        timeout=None,
    ):
        raise NotImplementedError

    def get_project(
        self,
        org_id,
        project_id,
        headers=None,
        timeout=None,
    ):
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    #################################################################
    # rest api parsers
    #################################################################
    def build_ltm_ctx(self, data, use_xml=None, do_json_str=None):
        raise NotImplementedError

    def build_stm_ctx(self, data, use_xml=None, do_json_str=None):
        raise NotImplementedError

    def build_stm_sum_ctx(self, data, use_xml=None, do_json_str=None):
        raise NotImplementedError

    def build_sm_ctx(self, data, use_xml=None, do_json_str=None):
        raise NotImplementedError

    def split_data(self, data):
        raise NotImplementedError
