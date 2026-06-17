"""8.7 Eval PPO — measure the tuned policy's average reward and compare it to the frozen SFT reference.

What this file does
-------------------
After PPO training, we want a single number that says "did the policy get better?".
This script is that scoreboard. It loads three things: the PPO-tuned policy, a
FROZEN SFT reference policy (the starting point, before any RL), and the Part 7
reward model (the judge). It then draws a small pool of prompts, lets BOTH policies
write a response to each, and asks the reward model to score the response. The
average reward over the pool is the metric. Higher tuned-vs-reference reward means
PPO actually pushed the policy toward answers the judge likes.

    for each prompt:
        prefix   = "Human: ... Assistant:"          # prompt text only
        y        = policy.generate(prefix)           # tuned take
        y_old    = ref.generate(prefix)              # frozen SFT take (for comparison)
        text     = format(prompt, policy_response)
        r        = reward_model(text)                # judge gives one number
    return mean(r over prompts)                      # the scoreboard

Note: this tiny script returns the mean reward of the TUNED policy. It also
generates the reference's take (y_old) on the same prompt so you can eyeball /
extend the comparison, but only the tuned response is fed to the reward model here.

Where this fits in the Part 8 RLHF-PPO loop
-------------------------------------------
    prompt
       |
    [ policy does a take: generate + record old_logp   (policy.py / rollout.py) ]
       |
    [ judge scores the take: reward model r            (part_7 RewardModel)     ]
       |
    [ KL leash: shaped_r = r - kl * (logp - ref_logp)  (vs frozen SFT reference)]
       |
    [ advantage = shaped_r - value baseline, normalized (train_ppo.py)          ]
       |
    [ PPO clipped update: min(unclipped, clipped)      (ppo_loss.py)            ]
       |
    [ AdamW step on the policy only                    (train_ppo.py)           ]
       |
    [ eval: avg reward, tuned policy vs frozen ref     (eval_ppo.py)            ]   <-- THIS FILE

Math
----
The only "math" here is the metric itself: the mean reward-model score over the
prompt pool.

    metric = (1 / N) * sum_{i=1..N} r_i

      N   = number of prompts sampled (n, default 16)
      r_i = reward_model(format(prompt_i, response_i)) , a single scalar per prompt
            (the RM's masked-mean-pooled sentence vector -> linear head -> one number)

Higher metric = the judge likes the policy's answers more. You read it relative
to the reference: tuned mean reward vs SFT mean reward.

Visualization
-------------
See notebook section 8.7 — plots the average RM reward of the tuned policy next to
the frozen SFT reference, showing whether PPO moved the score up.

Shapes
------
  ids          : list[int] length L     prompt prefix encoded to token ids
  x            : (1, T)                  one prompt row, last block_size ids, T <= block_size
  y / y_old    : (1, T + <=128)          generate() appends up to 128 new tokens to the prompt
  resp         : str                     decoded NEW tokens only (slice off the prompt)
  z            : (1, T')                  formatted "prompt+response" re-encoded, T' <= block_size
  rm(z)        : (1,)                     reward model: one scalar per row -> [0].item() = float r
  rewards      : list[float] length N     one reward per prompt; mean is the returned metric
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
        prefix = format_prompt_only(p).replace('</s>', '')  # "Human: ... Assistant:" prompt only
        ids = tok.encode(prefix)                             # list[int] length L
        x = torch.tensor([ids[-tok.block_size:]], dtype=torch.long, device=device)  # (1, T), T <= block_size
        with torch.no_grad():
            y = pol.generate(x, max_new_tokens=128, temperature=0.2, top_k=50)      # (1, T + <=128) tuned take
            y_old = ref.generate(x, max_new_tokens=128, temperature=0.2, top_k=50)  # (1, T + <=128) frozen SFT take
        resp = tok.decode(y[0].tolist()[len(ids[-tok.block_size:]):])      # decode NEW tokens only (drop prompt)
        resp_old = tok.decode(y_old[0].tolist()[len(ids[-tok.block_size:]):])  # reference response, for comparison

        # compute RM reward on formatted full text
        from part_6.formatters import Example, format_example
        text = format_example(Example(p, resp))             # stitch prompt+response into RM input string
        z = torch.tensor([tok.encode(text)[:tok.block_size]], dtype=torch.long, device=device)  # (1, T'), T' <= block_size
        with torch.no_grad():
            r = rm(z)[0].item()                             # rm(z): (1,) -> [0].item() = one scalar reward (float)
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