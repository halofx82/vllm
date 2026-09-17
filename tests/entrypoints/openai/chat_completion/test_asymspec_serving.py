# SPDX-License-Identifier: Apache-2.0
from vllm.entrypoints.openai.chat_completion.asymspec_serving import (
    RESERVED_EXTRA_ARG,
    has_reserved_extra_arg,
    select_recent_messages,
)


def test_asymspec_server_reserves_augmented_full_prompt_payload():
    assert has_reserved_extra_arg({RESERVED_EXTRA_ARG: [1, 2, 3]})
    assert not has_reserved_extra_arg({"unrelated": 1})


def test_asymspec_compression_keeps_pinned_and_recent_complete_turns():
    messages = [
        {"role": "system", "content": "pinned"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new"},
    ]

    assert select_recent_messages(messages, lambda candidate: len(candidate) <= 2) == [
        messages[0],
        messages[-1],
    ]
