"""Synchronous complete short tool trajectories with exact sampled action logprobs."""

import json

import torch

from minifrontier.chat_controls import parse_action, resolve
from minifrontier.inference import generate_ids
from minifrontier.training.tool_environment import ToolEnvironment


def sample_tool_trajectory(model, prompt, tokenizer, row, media, *, max_actions, context_limit):
    environment = ToolEnvironment(row["environment"])
    ids = prompt[0].tolist()
    labels = [-100] * len(ids)
    behavior = [0.0] * (len(ids) - 1)
    trace = []
    remaining, response, termination = max_actions, "", "length"
    _, effort = resolve(row.get("mode"), row["effort"])
    while remaining > 0:
        budget = min(remaining, context_limit - len(ids))
        if budget <= 0:
            termination = "context_limit"
            break
        generated, logp, mask = generate_ids(
            model,
            torch.tensor([ids], device=prompt.device),
            max_new_tokens=budget,
            vocab_size=tokenizer.get_vocab_size(),
            temperature=1.0,
            top_p=1.0,
            return_behavior=True,
            media=media,
        )
        actions = generated[0, len(ids) :].tolist()
        sampled_logp = logp[0, mask[0]].tolist()
        if len(actions) != len(sampled_logp) or not actions:
            raise RuntimeError("single-trajectory sampler returned inconsistent actions")
        ids += actions
        labels += actions
        behavior += sampled_logp
        remaining -= len(actions)
        parsed = parse_action(actions, tokenizer, require_eos=True)
        turn = dict(action_ids=actions, behavior_logp=sampled_logp, parsed=parsed)
        trace.append(turn)
        if parsed["kind"] == "final":
            response, termination = parsed["text"], "final"
            break
        if parsed["kind"] != "tool":
            termination = parsed["kind"]
            break
        observations = environment.execute(parsed["text"])
        turn["observations"] = observations
        # Observation framing is identical to encode(... role='tool'); the next
        # assistant chooses its own think/final/tool marker as a sampled action.
        context = [
            6,
            19,
            *tokenizer.encode(
                json.dumps(observations, ensure_ascii=False, sort_keys=True),
                add_special_tokens=False,
            ).ids,
            2,
            5,
            {"low": 12, "high": 13, "max": 14}[effort],
        ]
        if remaining == 0 or len(ids) + len(context) >= context_limit:
            termination = "length" if remaining == 0 else "context_limit"
            break
        ids += context
        labels += [-100] * len(context)
        behavior += [0.0] * len(context)
    return dict(
        ids=ids,
        labels=labels,
        behavior=behavior,
        response=response,
        termination=termination,
        environment=environment,
        trace=trace,
    )
