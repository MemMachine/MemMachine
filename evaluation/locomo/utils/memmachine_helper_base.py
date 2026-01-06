# utility for interacting with MemMachine
# There are 2 copies of this file, please keep them both the same
# 1. memmachine-test/benchmark/mem0_locomo/tests/memmachine
# 2. memmachine-test/benchmark/mem0_locomo/tests/mods/MemMachine/evaluation/locomo/utils
# ruff: noqa: PTH118, C901, RUF059, SIM108

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

    def edwin_timestamp_format(self, timestamp_str):
        """timestamp format that Edwin used
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

    def build_episodic_ctx(self, data, use_xml=None, do_summary=None):
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
            do_summary=do_summary,
            do_episodic=True,
            do_semantic=False,
        )

    def build_semantic_ctx(self, data, use_xml=None, do_summary=None):
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
            do_summary=do_summary,
            do_episodic=False,
            do_semantic=True,
        )

    def build_ctx(
        self, data, use_xml=None, do_summary=None, do_episodic=None, do_semantic=None
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
        if self.rest_variation == 1:
            ltm_ctx = self.build_ltm_ctx_v1(data, use_xml=use_xml)
            stm_ctx = self.build_stm_ctx_v1(data, use_xml=use_xml)
            stm_sum_ctx = self.build_stm_sum_ctx_v1(data, use_xml=use_xml)
            sm_ctx = self.build_sm_ctx_v1(data, use_xml=use_xml)
        elif self.rest_variation == 2:
            ltm_ctx = self.build_ltm_ctx_v2(data, use_xml=use_xml)
            stm_ctx = self.build_stm_ctx_v2(data, use_xml=use_xml)
            stm_sum_ctx = self.build_stm_sum_ctx_v2(data, use_xml=use_xml)
            sm_ctx = self.build_sm_ctx_v2(data, use_xml=use_xml)

        if ltm_ctx and do_episodic:
            ctx += ltm_ctx
            ctx += "\n"
        if stm_ctx and do_episodic:
            ctx += stm_ctx
            ctx += "\n"
        if stm_sum_ctx and do_summary:
            ctx += stm_sum_ctx
            ctx += "\n"
        if sm_ctx and do_semantic:
            ctx += sm_ctx
            ctx += "\n"
        return ctx

    def split_data(self, data):
        le = []  # long term memory episodes
        se = []  # short term memory episodes
        ss = []  # short term memory summaries
        sm = []  # semantic memory facts
        if not self.rest_variation:
            self.check_rest_variation(data)
        if self.rest_variation == 1:
            le, se, ss, sm = self.split_data_v1(data)
        elif self.rest_variation == 2:
            le, se, ss, sm = self.split_data_v2(data)
        return le, se, ss, sm

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

    ############################################################
    # v1 functions

    def build_ltm_ctx_v1(self, data, use_xml=None):
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
            metadata = episode["user_metadata"]
            if not metadata:
                metadata = {}
            if "source_timestamp" in metadata:
                ts = metadata["source_timestamp"]
            else:
                ts = episode["timestamp"]
            ts = self.edwin_timestamp_format(ts)
            if "source_speaker" in metadata:
                user = metadata["source_speaker"]
            else:
                user = episode["producer_id"]
            content = episode["content"]
            if not content:
                continue
            ctx = f"[{ts}] {user}: {content}"
            ltm_ctx += f"{ctx}\n"
        if ltm_ctx and use_xml:
            ltm_ctx = (
                f"<LONG TERM MEMORY EPISODES>\n{ltm_ctx}\n</LONG TERM MEMORY EPISODES>"
            )
        return ltm_ctx

    def build_stm_ctx_v1(self, data, use_xml=None):
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
            metadata = episode["user_metadata"]
            if not metadata:
                metadata = {}
            if "source_timestamp" in metadata:
                ts = metadata["source_timestamp"]
            else:
                ts = episode["timestamp"]
            ts = self.edwin_timestamp_format(ts)
            if "source_speaker" in metadata:
                user = metadata["source_speaker"]
            else:
                user = episode["producer_id"]
            content = episode["content"]
            if not content:
                continue
            ctx = f"[{ts}] {user}: {content}"
            stm_ctx += f"{ctx}\n"
        if stm_ctx and use_xml:
            stm_ctx = (
                f"<WORKING MEMORY EPISODES>\n{stm_ctx}\n</WORKING MEMORY EPISODES>"
            )
        return stm_ctx

    def build_stm_sum_ctx_v1(self, data, use_xml=None):
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
            stm_sum_ctx += f"{stm_summary}\n"
        if stm_sum_ctx and use_xml:
            stm_sum_ctx = (
                f"<WORKING MEMORY SUMMARY>\n{stm_sum_ctx}\n</WORKING MEMORY SUMMARY>"
            )
        return stm_sum_ctx

    def build_sm_ctx_v1(self, data, use_xml=None):
        if use_xml is None:
            use_xml = True
        if "content" not in data or "profile_memory" not in data["content"]:
            return ""
        sm_list = data["content"]["profile_memory"]
        sm_ctx = ""
        print(f"ERROR: @@@@@ not implemented yet sm_list={sm_list}")
        return sm_ctx

    def split_data_v1(self, data):
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

    ############################################################
    # v2 functions

    def build_ltm_ctx_v2(self, data, use_xml=None):
        if use_xml is None:
            use_xml = True
        if "content" not in data or "episodic_memory" not in data["content"]:
            return ""
        em = data["content"]["episodic_memory"]
        ltm_ctx = ""
        ltm = em["long_term_memory"]
        ltm_episodes = ltm["episodes"]
        for episode in ltm_episodes:
            metadata = episode["metadata"]
            if not metadata:
                metadata = {}
            if "source_timestamp" in metadata:
                ts = metadata["source_timestamp"]
            else:
                ts = episode["created_at"]
            ts = self.edwin_timestamp_format(ts)
            if "source_speaker" in metadata:
                user = metadata["source_speaker"]
            else:
                user = episode["producer_id"]
            content = episode["content"]
            if not content:
                continue
            ctx = f"[{ts}] {user}: {content}"
            ltm_ctx += f"{ctx}\n"
        if ltm_ctx and use_xml:
            ltm_ctx = (
                f"<LONG TERM MEMORY EPISODES>\n{ltm_ctx}\n</LONG TERM MEMORY EPISODES>"
            )
        return ltm_ctx

    def build_stm_ctx_v2(self, data, use_xml=None):
        if use_xml is None:
            use_xml = True
        if "content" not in data or "episodic_memory" not in data["content"]:
            return ""
        em = data["content"]["episodic_memory"]
        stm_ctx = ""
        stm = em["short_term_memory"]
        stm_episodes = stm["episodes"]
        for episode in stm_episodes:
            metadata = episode["metadata"]
            if not metadata:
                metadata = {}
            if "source_timestamp" in metadata:
                ts = metadata["source_timestamp"]
            else:
                ts = episode["created_at"]
            ts = self.edwin_timestamp_format(ts)
            if "source_speaker" in metadata:
                user = metadata["source_speaker"]
            else:
                user = episode["producer_id"]
            content = episode["content"]
            if not content:
                continue
            ctx = f"[{ts}] {user}: {content}"
            stm_ctx += f"{ctx}\n"
        if stm_ctx and use_xml:
            stm_ctx = (
                f"<WORKING MEMORY EPISODES>\n{stm_ctx}\n</WORKING MEMORY EPISODES>"
            )
        return stm_ctx

    def build_stm_sum_ctx_v2(self, data, use_xml=None):
        if use_xml is None:
            use_xml = True
        if "content" not in data or "episodic_memory" not in data["content"]:
            return ""
        em = data["content"]["episodic_memory"]
        stm_sum_ctx = ""
        stm = em["short_term_memory"]
        stm_summaries = stm["episode_summary"]
        for stm_summary in stm_summaries:
            if not stm_summary:
                continue
            stm_sum_ctx += f"{stm_summary}\n"
        if stm_sum_ctx and use_xml:
            stm_sum_ctx = (
                f"<WORKING MEMORY SUMMARY>\n{stm_sum_ctx}\n</WORKING MEMORY SUMMARY>"
            )
        return stm_sum_ctx

    def build_sm_ctx_v2(self, data, use_xml=None):
        if use_xml is None:
            use_xml = True
        if "content" not in data or "semantic_memory" not in data["content"]:
            return ""
        sm_list = data["content"]["semantic_memory"]
        sm_ctx = ""
        for sm_item in sm_list:
            feature = sm_item["feature_name"]
            value = sm_item["value"]
            if not feature or not value:
                continue
            sm_ctx += f"- {feature}: {value}\n"
        if sm_ctx and use_xml:
            sm_ctx = f"<SEMANTIC MEMORY>\n{sm_ctx}\n</SEMANTIC MEMORY>"
        return sm_ctx

    def split_data_v2(self, data):
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
