from memmachine.main.memmachine import MemoryType
from memmachine.server.api_v2.spec import ListMemoriesSpec


def test_list_memories_spec():
    spec = ListMemoriesSpec(
        org_id="test-org",
        project_id="test-project",
        page_size=50,
        page_num=10,
    )
    assert spec.type == MemoryType.Episodic
    assert spec.filter == ""
