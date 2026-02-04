# utility for interacting with MemMachine
# There are 2 copies of this file, please keep them both the same
# 1. memmachine-test/benchmark/mem0_locomo/tests/memmachine
# 2. memmachine-test/benchmark/mem0_locomo/tests/mods/MemMachine/evaluation/locomo/utils
# ruff: noqa: PTH118

import os
import sys

if True:
    # find path to other scripts and modules
    my_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.abspath(os.path.join(my_dir, ".."))
    top_dir = os.path.abspath(os.path.join(test_dir, ".."))
    utils_dir = os.path.join(top_dir, "utils")
    sys.path.insert(1, test_dir)
    sys.path.insert(1, top_dir)
    from memmachine_helper_restapiv1 import MemmachineHelperRestapiv1
    from memmachine_helper_restapiv2 import MemmachineHelperRestapiv2


class MemmachineHelper:
    """MemMachine helper factory
    Use this to create MemMachine helper instances
    """

    @staticmethod
    def factory(name, log=None, url=None):
        """Create a memmachine object with the specified interface name

        Currently name can be one of:
            restapiv2

        Inputs:
            name (str): can only be restapiv2

        Outputs:
            object: returns the request object
        """
        my_obj = None
        if name:
            name = name.upper()
        if name == "RESTAPIV1":
            my_obj = MemmachineHelperRestapiv1(log=log, url=url)
        elif name == "RESTAPIV2":
            my_obj = MemmachineHelperRestapiv2(log=log, url=url)
        else:
            raise AssertionError(f"ERROR: unknown name={name}")
        return my_obj
