"""9.7 GRPO Eval — average reward of the tuned policy vs the frozen SFT reference.

What this file does
-------------------
The headline check for RLHF: did tuning actually raise the judge's scores? This
script (still named `eval_ppo.py` -- a Part-8 leftover; GRPO reuses it unchanged)
generates from BOTH the GRPO-tuned policy and a frozen SFT reference on a small
prompt pool, scores each generation with the Part 7 reward model, and reports the
average. A positive (policy - reference) gap is the result of the whole pipeline.

    for each prompt:
        y      = policy.generate(prompt)         # the tuned model's answer
        y_old  = ref.generate(prompt)            # the frozen SFT reference's answer
        r      = reward_model(prompt + y)        # judge scores the tuned answer
    return mean(r)

Honest notes for readers:
  - The filename is `eval_ppo.py`, not `eval_grpo.py` -- GRPO inherits Part 8's eval verbatim.
  - The reference checkpoint path is HARD-CODED: "../part_6/runs/sft-demo/model_last.pt".
  - This tiny script averages reward of the TUNED policy only (y_old is generated but the
    returned mean uses the tuned policy's reward); it demonstrates the mechanism, not a
    rigorous A/B. The notebook 9.7 cell scores both sides and prints the delta.

Where this fits in the Part 9 RLHF-GRPO loop
--------------------------------------------
    prompt
       |
    [ policy does a GROUP of takes: G generations + old_logp  (policy.py / rollout.py) ]
       |
    [ judge scores each take: reward r_1..r_G                 (part_7 RewardModel)     ]
       |
    [ group baseline: A_i = r_i - mean(r over the group)      (train_grpo.py)          ]
       |
    [ clipped surrogate: min(unclipped, clipped)             (grpo_loss.py)           ]
       |
    [ + KL penalty in the loss: kl_coef * KL(new || ref)     (grpo_loss.py)           ]
       |
    [ AdamW step on the policy only (no value head)          (train_grpo.py)          ]
       |
    [ eval: avg reward, tuned policy vs frozen ref           (eval_ppo.py)            ]   <-- THIS FILE

Math
----
  avg_reward = (1/n) * sum_i  reward_model( format(prompt_i, policy.generate(prompt_i)) )

  Higher avg_reward => the policy earns more approval from the judge than before tuning.
  (This is also where reward hacking would show up: a high score on gibberish means the
  KL penalty in training was too weak.)

Visualization
-------------
See notebook section 9.7 — generate from both the tuned policy and the frozen reference,
score both with the reward model, and compare the means (the delta is the RLHF gain).

Shapes
------
  ids       : list[int]          encoded prompt-only tokens (clipped to block_size)
  x         : (1, P_i)           prompt tensor fed to generate
  y / y_old : (1, P_i + 128)     generated sequences (tuned policy / frozen reference)
  z         : (1, T)             formatted (prompt+response) tokenized for the reward model
  r         : float              scalar reward for one generation
  return    : float              mean reward over the n prompts
"""
from __future__ import annotations
import argparse, torch
from pathlib import Path

from policy import PolicyWithValue
from rollout import RLHFTokenizer, sample_prompts, format_prompt_only

# Reward model
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_7'))
from model_reward import RewardModel  # noqa: E402


def score_policy(policy_ckpt: str, rm_ckpt: str, bpe_dir: str | None, n: int = 16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = RLHFTokenizer(block_size=256, bpe_dir=bpe_dir)

    ckpt = torch.load(policy_ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    pol = PolicyWithValue(cfg.get('vocab_size', tok.vocab_size), cfg.get('block_size', tok.block_size),
                          cfg.get('n_layer', 2), cfg.get('n_head', 2), cfg.get('n_embd', 128)).to(device)
    pol.load_state_dict(ckpt['model'])
    pol.eval()

    # For comparing against reference policy (SFT)
    ref = PolicyWithValue(cfg.get('vocab_size', tok.vocab_size), cfg.get('block_size', tok.block_size),
                          cfg.get('n_layer', 2), cfg.get('n_head', 2), cfg.get('n_embd', 128)).to(device)
    ckpt_ref = torch.load("../part_6/runs/sft-demo/model_last.pt", map_location=device) # hardcoded path to SFT checkpoint
    ref.lm.load_state_dict(ckpt_ref['model']) 
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()

    rckpt = torch.load(rm_ckpt, map_location=device)
    rm = RewardModel(vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size), block_size=rckpt['config'].get('block_size', tok.block_size),
                     n_layer=rckpt['config'].get('n_layer', 4), n_head=rckpt['config'].get('n_head', 4), n_embd=rckpt['config'].get('n_embd', 256)).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    prompts = sample_prompts(n)
    rewards = []
    for p in prompts:
        prefix = format_prompt_only(p).replace('</s>', '')
        ids = tok.encode(prefix)
        x = torch.tensor([ids[-tok.block_size:]], dtype=torch.long, device=device)
        with torch.no_grad():
            y = pol.generate(x, max_new_tokens=128, temperature=0.2, top_k=50)
            y_old = ref.generate(x, max_new_tokens=128, temperature=0.2, top_k=50)
        resp = tok.decode(y[0].tolist()[len(ids[-tok.block_size:]):])
        resp_old = tok.decode(y_old[0].tolist()[len(ids[-tok.block_size:]):])

        # compute RM reward on formatted full text
        from part_6.formatters import Example, format_example
        text = format_example(Example(p, resp))
        z = torch.tensor([tok.encode(text)[:tok.block_size]], dtype=torch.long, device=device)
        with torch.no_grad():
            r = rm(z)[0].item()
        rewards.append(r)
    return sum(rewards)/max(1,len(rewards))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--policy_ckpt', type=str, required=True)
    p.add_argument('--reward_ckpt', type=str, required=True)
    p.add_argument('--split', type=str, default='val[:32]')  # unused in this tiny script
    p.add_argument('--bpe_dir', type=str, default=None)
    args = p.parse_args()

    avg_r = score_policy(args.policy_ckpt, args.reward_ckpt, args.bpe_dir, n=16)
    print(f"Avg RM reward: {avg_r:.4f}")