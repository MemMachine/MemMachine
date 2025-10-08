# test migration.py using locomo data
# 1. cd ~/MemMachine; ./memmachine-compose.sh  # start MemMachine
# 2. cd ~/mem-migration-hack; mkdir data  # create data dir
# 3. cp ~/MemMachine/evaluation/locomo/locomo10.json data  # copy locomo dataset into data dir
# 4. pytest  # run test

from migration import MigrationHack


def test_migration():
    base_url = "http://localhost:8080"
    chat_history = "data/locomo10.json"
    chat_type = "locomo"
    start_time = "0"
    max_messages = 0
    summarize = False
    summarize_every = 20
    migration_hack = MigrationHack(
        base_url=base_url,
        user_session_file="user_session.json",
        chat_history_file=chat_history,
        chat_type=chat_type,
        start_time=start_time,
        max_messages=max_messages,
    )

    migration_hack.migrate(summarize=summarize, summarize_every=summarize_every)
    print("== All completed successfully")


if __name__ == "__main__":
    test_migration()
