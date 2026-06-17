"""8.6 PPO Trainer — run one RLHF-PPO update loop that nudges the SFT policy toward higher reward.

What this file does
-------------------
This is the conductor for the whole Part 8 loop. It loads ONE Part 6 SFT
checkpoint twice: once as the trainable `policy` (which also grows a value
head) and once as a frozen `ref` (the reference we are not allowed to drift
too far from). It loads the Part 7 reward model, then repeats this per batch:

    for each step:
        prompts  = sample a few from the pool
        seq      = policy.generate(prompt)        # the policy "does a take"
        reward   = reward_model(prompt+response)  # one scalar at the LAST token
        old_logp = log prob the CURRENT policy gave each generated token
        ref_logp = log prob the FROZEN reference gives the same tokens
        old_val  = value head's baseline for each token
        kl       = old_logp - ref_logp            # per-token drift from ref
        shaped_r = reward - kl_coef * kl          # the "KL leash"
        adv      = (shaped_r - old_val), then normalized   # NO GAE here
        loss     = ppo_losses(...)                # clipped surrogate + value loss
        AdamW.step()                              # update the POLICY only

Note: this does exactly ONE gradient pass per collected batch (the inline
comment admits real PPO replays each batch several times). It is a teaching
demo, not a production trainer.

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
    [ AdamW step on the policy only                    (train_ppo.py)           ]   <-- THIS FILE
       |
    [ eval: avg reward, tuned policy vs frozen ref     (eval_ppo.py)            ]

Math
----
  KL leash (per action token, applied to the reward, NOT a separate loss):
      kl       = old_logp - ref_logp          # log pi_policy(a) - log pi_ref(a)
      shaped_r = reward - kl_coef * kl         # reward minus drift penalty
                 (kl_coef = 0.01; reward is non-zero only at the final token)

  Advantage (no GAE despite gamma/lam args being parsed):
      returns  = shaped_r                      # target value = immediate shaped reward
      adv      = returns - old_values          # how much better than the baseline
      adv      = (adv - adv.mean()) / adv.std().clamp_min(1e-6)   # normalized

  The clipped surrogate + value loss live in ppo_loss.py; this file just feeds it
  (new_logp, old_logp, adv, new_values, old_values, returns) with clip_ratio=0.2.

  Diagnostics (logged, not optimized):
      KL_move = mean(old_logp - new_logp)      # how far the update moved the policy
      KL_ref  = mean(new_logp - ref_logp)      # how far the policy now sits from ref

Visualization
-------------
See notebook section 8.6 — the reward-vs-step / KL-vs-step curves showing the
policy climbing reward while the KL leash keeps it close to the frozen reference.

Shapes
------
  in_ids[i]            : list[int] length P_i      prompt tokens for one sample
  out[0]               : (P_i + resp_len)          one generated sequence (prompt + response)
  seq                  : (B, T)                    padded batch, PAD id = 2; T = max_len <= block_size
  mask                 : (B, T) bool               True on RESPONSE (action) positions only
  rewards              : (B, T)                    scalar reward placed at the last real token only
  pol_lp / ref_lp      : (B, T-1)                  next-token logprob for seq[:,1:]
  values               : (B, T-1)                  value baseline, sliced to align with pol_lp
  act_mask = mask[:,1:]: (B, T-1) bool             which logprob positions are actions
  old_logp/ref_logp/
    old_values         : (N_act,)                  flattened over all action tokens in the batch
  kl, shaped_r, returns,
    adv                : (N_act,)                   one value per action token
  new_logp/new_values  : (N_act,)                  recomputed after the forward pass for the update
"""
from __future__ import annotations
import argparse, torch
from pathlib import Path

# import torch
# torch.manual_seed(0)

from policy import PolicyWithValue
from rollout import RLHFTokenizer, format_prompt_only, format_example, sample_prompts, gather_logprobs, shift_labels
from rollout import model_logprobs

# Reward model from Part 7
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_7'))
from model_reward import RewardModel  # noqa: E402

from ppo_loss import ppo_losses


def compute_reward(reward_model: RewardModel, tok: RLHFTokenizer, prompt: str, response: str, device) -> float:
    text = format_example(__import__('part_6.formatters', fromlist=['Example']).Example(prompt, response))
    ids = tok.encode(text)
    x = torch.tensor([ids[:tok.block_size]], dtype=torch.long, device=device)
    with torch.no_grad():
        r = reward_model(x)
    return float(r[0].item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='runs/ppo-demo')
    p.add_argument('--policy_ckpt', type=str, required=True, help='SFT checkpoint (Part 6)')
    p.add_argument('--reward_ckpt', type=str, required=True, help='Reward model checkpoint (Part 7)')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--resp_len', type=int, default=64)
    p.add_argument('--kl_coef', type=float, default=0.01)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--bpe_dir', type=str, default=None)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # tokenizer
    tok = RLHFTokenizer(block_size=args.block_size, bpe_dir=args.bpe_dir)

    # Load SFT policy as initial policy AND reference
    ckpt = torch.load(args.policy_ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    vocab_size = cfg.get('vocab_size', tok.vocab_size)
    block_size = cfg.get('block_size', tok.block_size)
    n_layer = cfg.get('n_layer', 2)
    n_head  = cfg.get('n_head', 2)
    n_embd  = cfg.get('n_embd', 128)

    policy = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    policy.lm.load_state_dict(ckpt['model'])  # initialize LM weights from SFT


    ref = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    ref.lm.load_state_dict(ckpt['model'])
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()

    # Reward model
    rckpt = torch.load(args.reward_ckpt, map_location=device)
    rm = RewardModel(vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size), block_size=rckpt['config'].get('block_size', tok.block_size),
                     n_layer=rckpt['config'].get('n_layer', 4), n_head=rckpt['config'].get('n_head', 4), n_embd=rckpt['config'].get('n_embd', 256)).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # small prompt pool
    prompts = sample_prompts(16)

    step = 0
    while step < args.steps:
        # ----- COLLECT ROLLOUT BATCH -----
        batch_prompts = prompts[ (step*args.batch_size) % len(prompts) : ((step+1)*args.batch_size) % len(prompts) ]
        if len(batch_prompts) < args.batch_size:
            batch_prompts += prompts[:args.batch_size-len(batch_prompts)]
        texts = [format_prompt_only(p).replace("</s>", "") for p in batch_prompts]
        in_ids = [tok.encode(t) for t in texts]  # list of list[int]; one prompt's tokens each

        with torch.no_grad():
            out_ids = []
            for i, x in enumerate(in_ids):
                idx = torch.tensor([x], dtype=torch.long, device=device)  # (1, P_i)
                out = policy.generate(idx, max_new_tokens=args.resp_len, temperature=0.2, top_k=3)  # (1, P_i+resp_len)
                out_ids.append(out[0].tolist())  # prompt + generated response, as list[int]

        # split prompt/response per sample
        data = []
        for i, prompt in enumerate(batch_prompts):
            full = out_ids[i]
            # find boundary: index where prompt ends in the tokenized form
            # Use original prompt tokenization length (clipped by block_size)
            p_ids = in_ids[i][-block_size:]
            boundary = len(p_ids)
            resp_ids = full[boundary:]
            # compute rewards via RM on formatted prompt+response text
            resp_text = tok.decode(resp_ids)
            r_scalar = compute_reward(rm, tok, prompt, resp_text, device)
            data.append((torch.tensor(full, dtype=torch.long), boundary, r_scalar))

        # pad to same length
        policy_ctx = getattr(policy, "block_size", block_size)
        max_len = min(policy_ctx, max(t[0].numel() for t in data))  # T (capped at block_size)
        B = len(data)
        seq = torch.zeros(B, max_len, dtype=torch.long, device=device)   # (B, T) padded token ids
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)  # (B, T) True on response tokens
        last_idx = torch.zeros(B, dtype=torch.long, device=device)       # (B,) index of each row's last real token
        rewards = torch.zeros(B, max_len, dtype=torch.float, device=device)  # (B, T) reward only at last token

        for i, (ids, boundary, r_scalar) in enumerate(data):
            L_full = ids.numel()
            L = min(L_full, max_len)
            drop = L_full - L                 # tokens dropped from the left
            b = max(0, boundary - drop)       # shift boundary after left-trim
            seq[i, :L] = ids[-L:]
            if L < max_len:
                seq[i, L:] = 2  # fill remaining positions with <pad> token
            mask[i, b:L] = True
            rewards[i, L-1] = r_scalar
            last_idx[i] = L-1


        # logprobs & values for policy and reference
        # model_logprobs returns (B, T-1) for next-token logp; align to seq[:,1:]
        pol_lp = model_logprobs(policy, seq)  # (B, T-1) logp the current policy gave seq[:,1:]
        ref_lp = model_logprobs(ref, seq)     # (B, T-1) logp the frozen reference gives the same
        # values for seq positions (B,T)
        with torch.no_grad():
            logits, values, _ = policy(seq, None)  # values: (B, T)
        values = values[:, :-1]  # (B, T-1) align to pol_lp

        # Select only action positions
        act_mask = mask[:,1:]  # (B, T-1); logprobs predict token t from <=t-1, so drop the first column
        old_logp = pol_lp[act_mask].detach()      # (N_act,) flattened over all action tokens
        ref_logp = ref_lp[act_mask].detach()      # (N_act,)
        old_values = values[act_mask].detach()    # (N_act,)

        # KL per action token and shaped rewards
        kl = (old_logp - ref_logp)                              # (N_act,) drift from reference
        shaped_r = rewards[:,1:][act_mask] - args.kl_coef * kl  # (N_act,) reward minus drift penalty

        # Compute advantages/returns with last‑step bootstrap = 0 (episodic per response)
        # Flatten by sequence order inside each sample; we’ll approximate by grouping tokens per sample using last_idx.
        # For tutorial simplicity, treat advantages = shaped_r - old_values (no GAE). Works for end-only reward.
        returns = shaped_r  # (N_act,) target value = immediate shaped reward
        adv = returns - old_values  # (N_act,) advantage = shaped reward above the value baseline
        # normalize adv
        adv = (adv - adv.mean()) / (adv.std().clamp_min(1e-6))  # (N_act,) zero-mean, unit-std

        # ----- UPDATE (single pass PPO for demo) -----
        # This step is done multiple times per batch in practice 
        policy.train()
        logits_new, values_new_full, _ = policy(seq, None)  # logits_new: (B, T, V), values_new_full: (B, T)
        logp_full = torch.log_softmax(logits_new[:, :-1, :], dim=-1)  # (B, T-1, V)
        labels = seq[:,1:]  # (B, T-1) the actual next token at each position
        new_logp_all = logp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1) logp of taken tokens
        new_logp = new_logp_all[act_mask]               # (N_act,) action-token logp after the update step
        new_values = values_new_full[:, :-1][act_mask]  # (N_act,) recomputed value baselines

        from ppo_loss import ppo_losses
        out_loss = ppo_losses(new_logp, old_logp, adv, new_values, old_values, returns,
                              clip_ratio=0.2, vf_coef=0.5, ent_coef=0.0)
        loss = out_loss.total_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        policy.eval()

        with torch.no_grad():
            # KL(old || new): movement of the updated policy from the snapshot used to collect data
            lp_post = model_logprobs(policy, seq)          # (B, T-1)
            lp_post = lp_post[act_mask]                    # only action positions
            kl_post = (old_logp - lp_post).mean()          # ≈ E[log π_old - log π_new]

            # KL(now || ref): how far the current policy is from the frozen reference
            lp_now = lp_post                               # already computed above on the same positions
            kl_ref_now = (lp_now - ref_logp).mean()        # ≈ E[log π_now - log π_ref]

        step += 1
        if step % 10 == 0:
            print(
                f"step {step} | loss {loss.item():.4f}"
                f"| value loss {out_loss.value_loss.item():.4f} | KL_move {kl_post.item():.6f} | KL_ref {kl_ref_now.item():.6f}"
            )


    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({'model': policy.state_dict(), 'config': {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
    }}, str(Path(args.out)/'model_last.pt'))
    print(f"Saved PPO policy to {args.out}/model_last.pt")

if __name__ == '__main__':
    main()