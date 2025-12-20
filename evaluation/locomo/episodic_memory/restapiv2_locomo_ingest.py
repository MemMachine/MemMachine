# ruff: noqa: PTH118, DTZ007, C901, DTZ005, G004, PERF403, UP031
import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timedelta

from dotenv import load_dotenv

if True:
    # find path to other scripts and modules
    my_dir = os.path.dirname(os.path.abspath(__file__))
    top_dir = os.path.abspath(os.path.join(my_dir, ".."))
    utils_dir = os.path.join(top_dir, "utils")
    sys.path.insert(1, utils_dir)
    sys.path.insert(1, top_dir)
    from memmachine_helper import MemmachineHelper


def datetime_from_locomo_time(locomo_time_str: str) -> datetime:
    return datetime.strptime(locomo_time_str, "%I:%M %p on %d %B, %Y")


def process_conversation(idx, item, args, mmai):
    if "conversation" not in item:
        return

    conversation = item["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    print(
        f"Processing conversation for group {idx} with speakers {speaker_a} and {speaker_b}..."
    )
    session_idx = 0
    timeout = 20 * args.batch_size

    while True:
        session_idx += 1
        session_id = f"session_{session_idx}"

        if session_id not in conversation:
            break

        try:
            session = conversation[session_id]
            try:
                ts_obj = datetime_from_locomo_time(
                    conversation[f"{session_id}_date_time"]
                )
            except Exception:
                ts_obj = datetime.now()
            ts_local_obj = ts_obj.astimezone()

            messages = []
            for message_index, message in enumerate(session):
                msg_ts_obj = ts_local_obj + message_index * timedelta(seconds=1)
                msg_ts_str = msg_ts_obj.isoformat()
                producer = message["speaker"]
                # get blip caption
                blip_caption = message.get("blip_caption")
                image_query = message.get("query")
                caption_str = ""
                if blip_caption and image_query:
                    caption_str = f" [Attached {blip_caption}: {image_query}]"
                else:
                    if blip_caption:
                        caption_str = f" [Attached {blip_caption}]"
                    elif image_query:
                        caption_str = f" [Attached a photo: {image_query}]"
                # content = text + blip caption
                content = message["text"] + caption_str

                # edwin_content = f'[{msg_ts_str}] {producer}: {content}'
                metadata = {
                    "locomo_session_id": session_id,
                    "source_timestamp": msg_ts_str,
                    "source_speaker": producer,
                }
                msg = {
                    "producer": producer,
                    # 'produced_for': 'test_agent',
                    # 'role': role,
                    "timestamp": msg_ts_str,
                    "content": content,
                    "metadata": metadata,
                }
                messages.append(msg)
                if len(messages) >= args.batch_size:
                    mmai.log.debug(f"add memory at index={message_index}")
                    mmai.add_memory(
                        messages=messages, mem_type=args.mem_type, timeout=timeout
                    )
                    messages = []
                    print(".", end="", flush=True)
            if messages:
                mmai.log.debug(f"add memory at index={message_index}")
                mmai.add_memory(
                    messages=messages, mem_type=args.mem_type, timeout=timeout
                )
                print(".", end="", flush=True)
            print()
        except Exception as ex:
            print()
            print(f"[{idx}]pc:ERROR:process conversation loop failed ex={ex}")
            print(f"[{idx}]pc:{traceback.format_exc()}")
        finally:
            print(f"[{idx}]pc:completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to the data file")
    # TOM1
    parser.add_argument(
        "--conv-start", type=int, default=1, help="start at this conversation"
    )
    parser.add_argument(
        "--conv-stop", type=int, default=1, help="stop at this conversation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="process this many messages per batch",
    )
    parser.add_argument("--mem-type", help="<episodic, semantic>, default is both")
    args = parser.parse_args()

    data_path = args.data_path

    with open(data_path, "r") as f:
        locomo_data = json.load(f)

    # TOM1
    mmai = MemmachineHelper.factory("restapiv2")
    health = mmai.get_health()
    print("mmai health:")
    for k, v in health.items():
        print(f"k={k} v={v}")

    metrics_before = mmai.get_metrics()
    with open(f"ingest_metrics_before_{os.getpid()}.json", "w") as fp:
        json.dump(metrics_before, fp, indent=4)

    print(f"batch_size={args.batch_size}")
    for idx, item in enumerate(locomo_data):
        conv_num = idx + 1
        if conv_num >= args.conv_start and conv_num <= args.conv_stop:
            process_conversation(idx, item, args, mmai)

    metrics_after = mmai.get_metrics()
    with open(f"ingest_metrics_after_{os.getpid()}.json", "w") as fp:
        json.dump(metrics_after, fp, indent=4)
    metrics_delta = mmai.diff_metrics()
    new_delta = {}
    for k, v in metrics_delta.items():
        if not k.endswith("_created"):
            new_delta[k] = v  # remove timestamps
    metrics_filename = f"ingest_metrics_delta_{os.getpid()}.json"
    with open(metrics_filename, "w") as fp:
        json.dump(new_delta, fp, indent=4)
    print(f"metrics_filename={metrics_filename}")
    i_tokens = 0
    o_tokens = 0
    e_tokens = 0
    if "language_model_openai_usage_input_tokens_total" in metrics_delta:
        i_tokens = int(metrics_delta["language_model_openai_usage_input_tokens_total"])
    if "language_model_openai_usage_output_tokens_total" in metrics_delta:
        o_tokens = int(metrics_delta["language_model_openai_usage_output_tokens_total"])
    if "embedder_openai_usage_prompt_tokens_total" in metrics_delta:
        e_tokens = int(metrics_delta["embedder_openai_usage_prompt_tokens_total"])
    tokens_str = (
        f"chat i_tokens={i_tokens} o_tokens={o_tokens} embedder tokens={e_tokens}"
    )
    print(f"save: memmachine {tokens_str}")
    vm_before = 0.0
    vm_after = 0.0
    rss_before = 0.0
    rss_after = 0.0
    if "process_virtual_memory_bytes" in metrics_before:
        vm_before = metrics_before["process_virtual_memory_bytes"]
    if "process_virtual_memory_bytes" in metrics_after:
        vm_after = metrics_after["process_virtual_memory_bytes"]
    if "process_resident_memory_bytes" in metrics_before:
        rss_before = metrics_before["process_resident_memory_bytes"]
    if "process_resident_memory_bytes" in metrics_after:
        rss_after = metrics_after["process_resident_memory_bytes"]
    vm_before /= 1073741824.0
    vm_after /= 1073741824.0
    rss_before /= 1073741824.0
    rss_after /= 1073741824.0
    mem_str = "VM_before "
    mem_str += "%8.4f GiB " % vm_before
    mem_str += "VM_after "
    mem_str += "%8.4f GiB " % vm_after
    mem_str += "RSS_before "
    mem_str += "%8.4f GiB " % rss_before
    mem_str += "RSS_after "
    mem_str += "%8.4f GiB " % rss_after
    print(f"save: memmachine memory {mem_str}")


if __name__ == "__main__":
    load_dotenv()
    main()
