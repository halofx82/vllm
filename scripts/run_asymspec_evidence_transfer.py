#!/usr/bin/env python3
"""Port frozen Step-51 external one-shot evidence transfer.

The carrier and final answer are deliberately separate requests.  This script
owns messages and rendering; the worker only writes the frozen-compatible
Top-2/L8 record requested by the carrier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.v1.spec_decode.asymspec.evidence_transfer import (
    evidence_wrapper,
    select_evidence,
)
from vllm.v1.spec_decode.asymspec.live_iteration import (
    ASYMSPEC_AUGMENTED_FULL_PROMPT_TOKEN_IDS,
)
from vllm.v1.spec_decode.asymspec.verifier_bridge import (
    ASYMSPEC_EXECUTION_MODE,
    ASYMSPEC_TARGET_ONLY_EXECUTION_MODE,
    EVIDENCE_CARRIER_OUTPUT_PATH,
)

VERIFIER = "Qwen/Qwen3.8-27B"
DRAFTER = "Qwen/Qwen3.5-4B"


def render(tokenizer, messages: list[dict], template: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=template,
        enable_thinking=False,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def usable_text(tokenizer, ids: list[int]) -> tuple[list[int], str]:
    kept: list[int] = []
    for token in ids:
        if token in tokenizer.all_special_ids:
            break
        kept.append(int(token))
    return kept, tokenizer.decode(kept).strip()


def enriched_messages(case: dict, evidence: str) -> list[dict]:
    messages = [dict(message) for message in case["messages"]]
    if not messages or messages[-1].get("role") != "user":
        raise RuntimeError(f"{case['id']}: expected final user question")
    messages.insert(-1, {"role": "user", "content": evidence_wrapper(evidence)})
    return messages


def build_llm(args: argparse.Namespace) -> LLM:
    return LLM(
        model=VERIFIER,
        runner="generate",
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=True,
        speculative_config={
            "method": "asymspec",
            "model": DRAFTER,
            "num_speculative_tokens": 2,
            "draft_tensor_parallel_size": args.tp,
            "asymspec_compressed_max_model_len": args.main_max_model_len,
            "asymspec_evidence_mode": "one_shot",
        },
    )


def sample(max_tokens: int, *, extra_args: dict | None = None) -> SamplingParams:
    return SamplingParams(
        temperature=0,
        seed=0,
        max_tokens=max_tokens,
        ignore_eos=True,
        extra_args=extra_args or {},
    )


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=66000)
    parser.add_argument("--main-max-model-len", type=int, default=9000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    args = parser.parse_args()
    if args.repetitions < 1 or args.max_tokens < 1:
        parser.error("--repetitions and --max-tokens must be positive")

    suite = json.loads(args.suite.read_text())
    if not isinstance(suite, list) or not suite:
        parser.error("suite must be a non-empty JSON list")
    if args.case_id:
        selected = set(args.case_id)
        suite = [case for case in suite if case.get("id") in selected]
        if selected - {case["id"] for case in suite}:
            parser.error("unknown --case-id")

    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="asymspec-evidence-") as temp:
        evidence_dir = Path(temp)
        llm = build_llm(args)
        tokenizer = llm.get_tokenizer()
        reference = AutoTokenizer.from_pretrained(VERIFIER)
        template = reference.chat_template
        if not isinstance(template, str):
            raise RuntimeError("Qwen reference chat template is unavailable")
        for index, case in enumerate(suite):
            compressed_ids = list(
                case.get("prompt_token_ids")
                or render(tokenizer, case["messages"], template)
            )
            full_ids = list(
                case.get("full_prompt_token_ids")
                or render(
                    tokenizer, case.get("full_messages", case["messages"]), template
                )
            )
            runs: list[dict] = []
            for repetition in range(args.repetitions):
                carrier_name = (
                    f"evidence-{index * args.repetitions + repetition:03d}.json"
                )
                carrier_file = evidence_dir / carrier_name
                carrier_start = time.perf_counter()
                llm.generate(
                    [{"prompt_token_ids": compressed_ids}],
                    sample(
                        1,
                        extra_args={
                            ASYMSPEC_AUGMENTED_FULL_PROMPT_TOKEN_IDS: full_ids,
                            EVIDENCE_CARRIER_OUTPUT_PATH: str(carrier_file),
                        },
                    ),
                    use_tqdm=False,
                )
                carrier_seconds = time.perf_counter() - carrier_start
                if not carrier_file.exists():
                    raise RuntimeError(f"{case['id']}: missing native carrier record")
                carrier = json.loads(carrier_file.read_text())
                evidence = select_evidence(
                    carrier["rollouts"],
                    full_top2=carrier["full_top2"],
                    full_source_ids=full_ids,
                    base_source_ids=compressed_ids,
                )
                raw_ids = [int(token) for token in evidence.selected_token_ids]
                visible_ids, evidence_text = usable_text(tokenizer, raw_ids)
                if not evidence_text:
                    raise RuntimeError(
                        f"{case['id']}: selected evidence has no visible text"
                    )
                if evidence.attribution_pass:
                    final_messages = enriched_messages(case, evidence_text)
                    final_ids = render(tokenizer, final_messages, template)
                    final_extra = {
                        ASYMSPEC_EXECUTION_MODE: ASYMSPEC_TARGET_ONLY_EXECUTION_MODE
                    }
                    fallback = False
                else:
                    final_ids = compressed_ids
                    final_extra = {ASYMSPEC_AUGMENTED_FULL_PROMPT_TOKEN_IDS: full_ids}
                    fallback = True
                target_start = time.perf_counter()
                output = llm.generate(
                    [{"prompt_token_ids": final_ids}],
                    sample(args.max_tokens, extra_args=final_extra),
                    use_tqdm=False,
                )[0]
                target_seconds = time.perf_counter() - target_start
                runs.append(
                    {
                        "carrier_seconds": carrier_seconds,
                        "target_seconds": target_seconds,
                        "total_seconds": carrier_seconds + target_seconds,
                        "target_prompt_tokens": len(final_ids),
                        "target_prefill_passes": 2,
                        "evidence": evidence.json(),
                        "evidence_visible_token_ids": visible_ids,
                        "evidence_text": evidence_text,
                        "fallback": fallback,
                        "generated_token_ids": list(output.outputs[0].token_ids),
                        "generated_text": output.outputs[0].text,
                    }
                )
            first = runs[0]
            if any(
                run["generated_token_ids"] != first["generated_token_ids"]
                for run in runs[1:]
            ):
                raise RuntimeError(f"{case['id']}: deterministic repetitions diverged")
            results.append(
                {
                    "id": case["id"],
                    "expected": case.get("expected"),
                    "position_bucket": case.get("position_bucket"),
                    "full_prompt_tokens": len(full_ids),
                    "compressed_prompt_tokens": len(compressed_ids),
                    "runs": runs,
                    "carrier_seconds_median": median(
                        [run["carrier_seconds"] for run in runs]
                    ),
                    "target_seconds_median": median(
                        [run["target_seconds"] for run in runs]
                    ),
                    "total_seconds_median": median(
                        [run["total_seconds"] for run in runs]
                    ),
                    "generated_text": first["generated_text"],
                    "generated_token_ids": first["generated_token_ids"],
                    "evidence": first["evidence"],
                    "evidence_text": first["evidence_text"],
                    "fallback": first["fallback"],
                    "answer_correct": case.get("expected", "")
                    in first["generated_text"],
                    "prompt_sha256": hashlib.sha256(
                        json.dumps(compressed_ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "mode": "one_shot_evidence_transfer",
                "evidence_transfer": True,
                "min_cf": 0.10,
                "min_margin": 0.05,
                "wrapper_placement": "prior-user",
                "target_prefill_passes": 2,
                "cases": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "correct": sum(r["answer_correct"] for r in results),
                "cases": len(results),
                "fallbacks": sum(r["fallback"] for r in results),
            }
        )
    )


if __name__ == "__main__":
    main()
