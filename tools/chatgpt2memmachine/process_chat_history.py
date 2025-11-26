import argparse
import datetime
import json
import re
import sys
import traceback


def timestamp_compare(ts1, ts2):
    ts1 = timestamp_ms_to_sec(ts1)
    ts2 = timestamp_ms_to_sec(ts2)
    if ts1 < ts2:
        return -1
    if ts1 > ts2:
        return 1
    return 0


def timestamp_ms_to_sec(ts):
    if isinstance(ts, float):
        ts = int(ts)
    if ts > 9999999999:
        ts = int(ts / 1000)
    return ts


def timestamp_to_obj(ts):
    ts = timestamp_ms_to_sec(ts)
    t_obj = datetime.datetime.fromtimestamp(ts)
    return t_obj


def locomo_count_conversations(infile, verbose=False):
    if verbose:
        print(f"lcc: loading locomo input file {infile}", file=sys.stderr)
    with open(infile) as fp:
        data = json.load(fp)
    # loop to load every session
    conv_count = 0
    section_count = 0
    done = False
    while not done:
        for section in data:
            section_count += 1
            if verbose:
                print(
                    f"lcc: look for next conversation section {section_count}",
                    file=sys.stderr,
                )
                # print(f'lcc: keys={list(section.keys())}', file=sys.stderr)
            if "conversation" in section:
                conv_count += 1
                if verbose:
                    print(f"lcc: found conversation {conv_count}", file=sys.stderr)
            if verbose:
                print(f"lcc: finished conversation {conv_count} (2)", file=sys.stderr)
            if done:
                break
        if verbose:
            print(f"lcc: counted all sections={section_count}", file=sys.stderr)
        done = True
    return conv_count


def load_locomo(
    infile, start_time=None, conv_num=None, max_messages=None, verbose=False
):
    if not start_time:
        start_time = 0
    if not conv_num:
        conv_num = 0
    if not max_messages:
        max_messages = 0
    lines = []
    if verbose:
        print(
            f"ll: start_time={start_time} conv_num={conv_num} max_messages={max_messages}",
            file=sys.stderr,
        )
    if verbose:
        print(f"ll: loading locomo input file {infile}", file=sys.stderr)
    with open(infile) as fp:
        data = json.load(fp)
    # loop to load every session
    conv_count = 0
    msg_count = 0
    section_count = 0
    done = False
    while not done:
        for section in data:
            section_count += 1
            if verbose:
                print(
                    f"ll: look for next conversation section {section_count}",
                    file=sys.stderr,
                )
                # print(f'll: keys={list(section.keys())}', file=sys.stderr)
            if "conversation" in section:
                conversation = section["conversation"]
                conv_count += 1
                if conv_num and conv_count != conv_num:
                    # user asked to do one specific conversation
                    continue
                if verbose:
                    print(f"ll: loading conversation {conv_count}", file=sys.stderr)
                for num in range(1, 9999):
                    session_name = f"session_{num}"
                    session_date_name = f"session_{num}_date_time"
                    if session_name not in conversation:
                        # processed all of the sessions in this conversation
                        if verbose:
                            print(
                                f"ll: finished conversation {conv_count} (1)",
                                file=sys.stderr,
                            )
                        break
                    if verbose:
                        print(
                            f"ll: loading conversation {conv_count} session {num}",
                            file=sys.stderr,
                        )
                    messages = conversation[session_name]
                    session_date_str = ""
                    session_date_obj = None
                    if not session_date_obj:
                        try:
                            session_date_str = conversation[session_date_name]
                            session_date_obj = datetime.datetime.strptime(
                                session_date_str, "%I:%M %p on %d %b, %Y"
                            )
                        except Exception:
                            pass
                    if not session_date_obj:
                        try:
                            session_date_str = conversation[session_date_name]
                            session_date_obj = datetime.datetime.strptime(
                                session_date_str, "%I:%M %p on %d %B, %Y"
                            )
                        except Exception:
                            pass
                    try:
                        session_time = session_date_obj.timestamp()
                        if start_time:
                            if timestamp_compare(start_time, session_time) > 0:
                                if verbose:
                                    print(
                                        f"ll: skipping old conversation {conv_count} session {num} time={session_time}",
                                        file=sys.stderr,
                                    )
                                break
                    except Exception:
                        if verbose:
                            print(
                                f"ll: ERROR: cannot read timestamp of conversation {conv_count} session {num} date={session_date_str}",
                                file=sys.stderr,
                            )
                    for message in messages:
                        if "text" in message:
                            lines.append(message["text"])
                            msg_count += 1
                            if max_messages and msg_count >= max_messages:
                                # user asked to do this many messages only
                                if verbose:
                                    print(
                                        f"ll: processed max messages={msg_count}",
                                        file=sys.stderr,
                                    )
                                done = True
                                break
                    if done:
                        break
                if verbose:
                    print(
                        f"ll: finished conversation {conv_count} sessions={num}",
                        file=sys.stderr,
                    )
            if verbose:
                print(f"ll: finished conversation {conv_count} (2)", file=sys.stderr)
            if done:
                break
        if verbose:
            print(f"ll: loaded all sections={section_count}", file=sys.stderr)
        done = True
    return lines


def openai_count_conversations(infile, verbose=False):
    if verbose:
        print(f"occ: loading openai input file {infile}", file=sys.stderr)
    with open(infile) as fp:
        data = json.load(fp)
    # loop to load every chat
    chat_count = 0
    for chat in data:
        chat_count += 1
    return chat_count


def load_openai(
    infile,
    start_time=None,
    conv_num=None,
    max_messages=None,
    verbose=False,
    chat_title=None,
):
    if not start_time:
        start_time = 0
    if not conv_num:
        conv_num = 0
    if not max_messages:
        max_messages = 0
    lines = []
    if verbose:
        print(
            f"lo: start_time={start_time} max_messages={max_messages}", file=sys.stderr
        )
    if verbose:
        print(f"lo: loading openai input file {infile}", file=sys.stderr)
    with open(infile) as fp:
        data = json.load(fp)
    # loop to load every chat
    chat_count = 0
    msg_count = 0
    done = False
    for chat in data:
        # load one chat into chat_data
        chat_count += 1
        if conv_num and chat_count != conv_num:
            # user asked to do one specific conversation
            continue
        # check title
        chat_title_actual = chat["title"]
        if chat_title and chat_title.lower() != chat_title_actual.lower():
            if verbose:
                print(f"lo: skipping chat title={chat_title_actual}", file=sys.stderr)
            continue
        # check time
        chat_time = chat["create_time"]
        # print(f'TOM1: chat_time={chat_time}')
        if start_time and timestamp_compare(start_time, chat_time) > 0:
            if verbose:
                print(
                    f"lo: skipping old chat {chat_count} time={chat_time}",
                    file=sys.stderr,
                )
            continue
        # load messages
        if verbose:
            print(f"lo: loading chat title={chat_title_actual}", file=sys.stderr)
        chat_data = []
        for id, chat_map in chat["mapping"].items():
            # validate
            if "message" not in chat_map:
                continue
            if not chat_map["message"]:
                continue
            message = chat_map["message"]
            if "author" not in message:
                continue
            if not message["author"]:
                continue
            if "role" not in message["author"]:
                continue
            if "content" not in message:
                continue
            if not message["content"]:
                continue
            if "content_type" not in message["content"]:
                continue
            try:
                msg_author = message["author"]
                msg_role = msg_author["role"]
                msg_ts = message["create_time"]
                msg_content = message["content"]
                msg_type = msg_content["content_type"]
                if msg_role == "user" and msg_type == "text":
                    msg_str = "".join(msg_content["parts"])
                    if not msg_ts:
                        if verbose:
                            print(
                                f"lo: ERROR: chat {chat_count} user message {msg_str} has no timestamp",
                                file=sys.stderr,
                            )
                    else:
                        datapoint = {
                            "timestamp": msg_ts,
                            "text": msg_str,
                        }
                        chat_data.append(datapoint)
            except Exception as ex:
                if verbose:
                    print(
                        f"lo: ERROR: processing chat message={message} ex={ex}",
                        file=sys.stderr,
                    )
                    print(traceback.format_exc(), file=sys.stderr)
        # sort messages
        chat_sorted = sorted(chat_data, key=lambda x: x["timestamp"])
        # save messages
        for message in chat_sorted:
            lines.append(message["text"])
            msg_count += 1
            if max_messages and msg_count >= max_messages:
                # user asked to do this many messages only
                if verbose:
                    print(f"lo: processed max messages={msg_count}", file=sys.stderr)
                done = True
                break
        if verbose:
            print(f"lo: finished chat title={chat_title_actual}", file=sys.stderr)
        if done:
            break
    return lines


def get_args():
    parser = argparse.ArgumentParser(description="Process chat history")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print debug info if available"
    )
    parser.add_argument(
        "-s",
        "--src",
        action="store",
        default="locomo",
        help="openai|locomo, default=locomo",
    )
    parser.add_argument("-i", "--infile", action="store", help="input chat history")
    parser.add_argument("-o", "--outfile", action="store", help="output parsed chat")
    parser.add_argument(
        "-t",
        "--start_time",
        action="store",
        help="only read messages after this time either YYYY-MM-DDTHH:MM:SS or secs since epoch",
    )
    parser.add_argument(
        "-n",
        "--max_messages",
        action="store",
        type=int,
        default=0,
        help="only read this many messages",
    )
    parser.add_argument("--openai_chat", action="store", help="load only this chat")
    parser.add_argument(
        "--conversation",
        action="store",
        type=int,
        default=0,
        help="load only this conversation",
    )
    parser.add_argument(
        "--how_many_conversations",
        action="store_true",
        help="returns how many conversations are in the input file",
    )
    # parser.add_argument('--summarize_every', action='store', type=int, default=0, help='summarize before storing into memmachine, default is 0')
    args = parser.parse_args()
    if not args.infile:
        print("ERROR: must specify --infile", file=sys.stderr)
        sys.exit(1)
    if args.start_time:
        ts = 0
        try:
            # time in int
            ts = int(args.start_time)
        except Exception:
            pass
        if not ts:
            try:
                # time is str
                time_obj = datetime.datetime.strptime(
                    args.start_time, "%Y-%m-%dT%H:%M:%S"
                )
                ts = time_obj.timestamp()
            except Exception:
                pass
        args.start_time = ts
    return args


if __name__ == "__main__":
    args = get_args()
    args.src = args.src.lower()
    lines = []
    if args.src == "locomo":
        if args.how_many_conversations:
            count = locomo_count_conversations(args.infile, args.verbose)
            lines = [f"{count}"]
        else:
            lines = load_locomo(
                args.infile,
                args.start_time,
                args.conversation,
                args.max_messages,
                args.verbose,
            )
    elif args.src == "openai":
        if args.how_many_conversations:
            count = openai_count_conversations(args.infile, args.verbose)
            lines = [f"{count}"]
        else:
            lines = load_openai(
                args.infile,
                args.start_time,
                args.conversation,
                args.max_messages,
                args.verbose,
                args.openai_chat,
            )
    else:
        print(f"ERROR: unknown input source {args.src}", file=sys.stderr)
        sys.exit(1)

    # save output
    if args.outfile:
        fp = open(args.outfile, "w")
    else:
        fp = sys.stdout
    for line in lines:
        line = line.strip()
        line = re.sub(r"\\n", " ", line)
        line = re.sub(r"\n", " ", line)
        print(f"{line}", file=fp)
