"""9.6 GRPO Trainer — run one RLHF-GRPO update loop that nudges the SFT policy toward higher reward.

What this file does
-------------------
This is Part 8's `train_ppo.py` with two edits: generate a GROUP of completions per
prompt, and replace the critic baseline with the group's average reward. It loads ONE
Part 6 SFT checkpoint twice (trainable `policy` + frozen `ref`), loads the Part 7
reward model, then repeats this per step:

    for each step:
        prompts  = P distinct prompts from the pool
        for each prompt, for g in range(G):        # a GROUP of G takes per prompt
            seq    = policy.generate(prompt)        # the policy "does a take"
            reward = reward_model(prompt+response)  # one scalar PER TAKE (trajectory-level)
        # ----- GROUP BASELINE (the GRPO heart) -----
        group_mean[i] = mean reward over the takes sharing prompt i
        traj_adv      = raw_rewards - group_mean    # A_i = r_i - group average
        adv_flat      = broadcast traj_adv to each take's action tokens, then normalize
        # ----- POLICY-ONLY UPDATE -----
        new_logp = log prob the CURRENT policy gives each action token
        kl_ref   = mean(new_logp - ref_logp)        # KL(new || ref) on action tokens
        loss     = ppo_policy_only_losses(new_logp, old_logp, adv_flat,
                                          kl_coef, kl_mean=kl_ref)  # clipped surrogate + KL penalty
        AdamW.step()                                # update the POLICY only (value head ignored)

Key differences from train_ppo.py:
  - B = P * G trajectories (a group of G per prompt), not one per prompt.
  - The baseline is the GROUP MEAN reward, not the value head. The value head is never read:
    `logits_new, _, _ = policy(seq, None)` discards `values`.
  - The KL enters via the LOSS (kl_coef * KL(new||ref)), not by shaping the reward.

Two honest notes for readers:
  - `kl_tok = old_logp - ref_logp` is computed and a comment mentions "trajectory-level KL
    shaping", but that line is NEVER USED -- the advantage is pure (reward - group mean). KL
    affects training only through the loss penalty. A vestigial leftover.
  - The advantage is normalized by subtracting the group mean and then z-normalizing the whole
    batch (global std). Canonical GRPO often divides each group by its OWN std; this build
    subtracts the group mean then normalizes globally -- same spirit, slightly different math.

Where this fits in the Part 9 RLHF-GRPO loop
--------------------------------------------
    prompt
       |
    [ policy does a GROUP of takes: G generations + old_logp  (policy.py / rollout.py) ]
       |
    [ judge scores each take: reward r_1..r_G                 (part_7 RewardModel)     ]
       |
    [ group baseline: A_i = r_i - mean(r over the group)      (train_grpo.py)          ]   <-- THIS FILE
       |
    [ clipped surrogate: min(unclipped, clipped)             (grpo_loss.py)           ]
       |
    [ + KL penalty in the loss: kl_coef * KL(new || ref)     (grpo_loss.py)           ]
       |
    [ AdamW step on the policy only (no value head)          (train_grpo.py)          ]   <-- THIS FILE
       |
    [ eval: avg reward, tuned policy vs frozen ref           (eval_ppo.py)            ]

Math
----
  Group baseline (the defining GRPO step; G = group_size completions sharing a prompt):
      group_mean_i = (1/|G|) * sum_{j in group(i)} reward_j      # the "curve" for that prompt
      traj_adv_i   = reward_i - group_mean_i                      # beat-the-group-average
      adv_flat     = traj_adv broadcast to each take's action tokens
      adv_flat     = (adv_flat - adv_flat.mean()) / adv_flat.std().clamp_min(1e-6)   # normalized
    Advantages within a group sum to ~0 (we subtracted the mean): always something to push
    up and something to push down. NO value head, NO GAE -- one advantage per take.

  KL penalty (applied in the loss, not the reward):
      kl_ref = mean(new_logp - ref_logp)        # KL(pi_new || pi_ref) over action tokens
      total  = policy_loss + kl_coef * kl_ref    # kl_coef = 0.01

  The clipped surrogate itself lives in grpo_loss.py; this file feeds it
  (new_logp, old_logp, adv_flat) with clip_ratio=0.2 and the kl_mean above.

  Diagnostics (logged, not optimized):
      KL_move = mean(old_logp - lp_post)         # how far the update moved the policy
      KL_ref  = mean(lp_post - ref_logp)         # how far the policy now sits from ref

Visualization
-------------
See notebook section 9.6 — one full GRPO step on tiny fresh models: a group of takes,
their group-mean baseline, the resulting advantages, the policy-only clipped+KL loss,
and the AdamW update (policy moves; ref + reward model stay frozen).

Shapes
------
  prompt_in_ids[i]     : list[int]                 prompt tokens for prompt i
  seq_list[k]          : (L_k,)                     one generated sequence (prompt + response)
  prompt_id_of[k]      : int in [0, P)              which prompt (group) trajectory k belongs to
  raw_rewards[k]       : float                      reward model score for take k (trajectory-level)
  seq                  : (B, T)                     padded batch, PAD id = 2; B = P*G, T <= block_size
  mask                 : (B, T) bool                True on RESPONSE (action) positions only
  act_mask = mask[:,1:]: (B, T-1) bool              which logprob positions are actions
  old_logp / ref_logp  : (N_act,)                   flattened over all action tokens (snapshot, no grad)
  group_mean           : (B,)                       each take's group-average reward
  traj_adv             : (B,)                       reward - group_mean, one per take
  adv_flat             : (N_act,)                   traj_adv broadcast to action tokens, normalized
  new_logp             : (N_act,)                   recomputed WITH grad for the update
  loss                 : ()                         scalar from ppo_policy_only_losses(...).total_loss
"""
# train_grpo.py
from __future__ import annotations
import argparse, torch
from pathlib import Path

from policy import PolicyWithValue  # we will ignore the value head
from rollout import RLHFTokenizer, format_prompt_only, sample_prompts, model_logprobs

# Reward model from Part 7
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_7'))
from model_reward import RewardModel  # noqa: E402

from grpo_loss import ppo_policy_only_losses


@torch.no_grad()
def compute_reward(reward_model: RewardModel, tok: RLHFTokenizer, prompt_text: str, response_ids: list[int], device) -> float:
    # Build full formatted text (as in your PPO)
    from part_6.formatters import Example, format_example
    resp_text = tok.decode(response_ids)
    text = format_example(Example(prompt_text, resp_text))
    ids = tok.encode(text)
    x = torch.tensor([ids[:tok.block_size]], dtype=torch.long, device=device)
    r = reward_model(x)
    return float(r[0].item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='runs/grpo-demo')
    p.add_argument('--policy_ckpt', type=str, required=True, help='SFT checkpoint (Part 6)')
    p.add_argument('--reward_ckpt', type=str, required=True, help='Reward model checkpoint (Part 7)')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch_prompts', type=int, default=32, help='number of distinct prompts per step (before grouping)')
    p.add_argument('--group_size', type=int, default=4, help='completions per prompt')
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--resp_len', type=int, default=64)
    p.add_argument('--kl_coef', type=float, default=0.01)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--bpe_dir', type=str, default=None)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # tokenizer
    tok = RLHFTokenizer(block_size=args.block_size, bpe_dir=args.bpe_dir)

    # Load SFT policy (and a frozen reference)
    ckpt = torch.load(args.policy_ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    vocab_size = cfg.get('vocab_size', tok.vocab_size)
    block_size = cfg.get('block_size', tok.block_size)
    n_layer = cfg.get('n_layer', 2)
    n_head  = cfg.get('n_head', 2)
    n_embd  = cfg.get('n_embd', 128)

    policy = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    policy.lm.load_state_dict(ckpt['model'])
    policy.eval()

    ref = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    ref.lm.load_state_dict(ckpt['model'])
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()

    # Reward model
    rckpt = torch.load(args.reward_ckpt, map_location=device)
    rm = RewardModel(vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size),
                     block_size=rckpt['config'].get('block_size', tok.block_size),
                     n_layer=rckpt['config'].get('n_layer', 4),
                     n_head=rckpt['config'].get('n_head', 4),
                     n_embd=rckpt['config'].get('n_embd', 256)).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # small prompt pool (reuse your helper)
    prompts_pool = sample_prompts(16)

    step = 0
    pool_idx = 0
    G = args.group_size

    while step < args.steps:
        # ----- SELECT PROMPTS -----
        # Choose P prompts, each will yield G completions → B = P*G trajectories
        P = max(1, args.batch_prompts)
        if pool_idx + P > len(prompts_pool):
            pool_idx = 0
        batch_prompts = prompts_pool[pool_idx: pool_idx + P]
        pool_idx += P

        # Tokenize prompt-only texts
        prompt_texts = [format_prompt_only(p).replace("</s>", "") for p in batch_prompts]
        prompt_in_ids = [tok.encode(t) for t in prompt_texts]

        # ----- GENERATE G COMPLETIONS PER PROMPT -----
        # We will collect all trajectories flat, but track their group/prompt ids.
        seq_list = []        # list[Tensor of token ids]
        boundary_list = []   # index where response starts in the (possibly clipped) sequence
        prompt_id_of = []    # which prompt this trajectory belongs to (0..P-1)
        raw_rewards = []     # scalar reward per trajectory (before KL shaping)
        last_idx_list = []   # for padding bookkeeping

        with torch.no_grad():
            for pid, p_ids in enumerate(prompt_in_ids):
                for g in range(G):
                    idx = torch.tensor([p_ids], dtype=torch.long, device=device)
                    out = policy.generate(idx, max_new_tokens=args.resp_len, temperature=2, top_k=3)
                    full_ids = out[0].tolist()

                    # split prompt/response
                    boundary = len(p_ids[-block_size:])  # prompt length clipped to context
                    resp_ids = full_ids[boundary:]
                    r_scalar = compute_reward(rm, tok, batch_prompts[pid], resp_ids, device)

                    seq_list.append(torch.tensor(full_ids, dtype=torch.long))
                    boundary_list.append(boundary)
                    prompt_id_of.append(pid)
                    raw_rewards.append(r_scalar)

        # ----- PAD TO BATCH -----
        B = len(seq_list)  # B = P*G
        policy_ctx = getattr(policy, "block_size", block_size)
        max_len = min(policy_ctx, max(s.numel() for s in seq_list))
        seq = torch.zeros(B, max_len, dtype=torch.long, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        last_idx = torch.zeros(B, dtype=torch.long, device=device)

        # keep a per-traj “action positions” mask and response-only boundary
        for i, (ids, bnd) in enumerate(zip(seq_list, boundary_list)):
            L_full = ids.numel()
            L = min(L_full, max_len)
            drop = L_full - L
            b = max(0, bnd - drop)  # shifted boundary after left-trim
            seq[i, :L] = ids[-L:]
            if L < max_len:
                seq[i, L:] = 2  # pad token
            # actions are predicting token t from <=t-1 → positions [1..L-1]
            # but we only care about response tokens: mask [b..L-1] → actions [b+1..L-1]
            mask[i, b:L] = True
            last_idx[i] = L - 1

        # ----- LOGPROBS & KL VS REF (token-level) -----
        # model_logprobs returns log p(x[t] | x[:t-1]) for t=1..T-1 over labels=seq[:,1:]
        with torch.no_grad():
            pol_lp_full = model_logprobs(policy, seq)  # (B, T-1)
            ref_lp_full = model_logprobs(ref, seq)     # (B, T-1)

        # action positions (predict positions [1..T-1]); we want only response tokens:
        act_mask = mask[:, 1:]  # align to (B, T-1)
        old_logp = pol_lp_full[act_mask].detach()
        ref_logp = ref_lp_full[act_mask].detach()

        # per-token KL on action tokens
        kl_tok = (old_logp - ref_logp)  # (N_act,)

        # ----- SHAPED TRAJECTORY REWARD & GROUP BASELINE -----
        # For GRPO, advantage is trajectory-level and broadcast to its tokens.
        # We include KL shaping at trajectory level using mean token KL per trajectory.
        # First, compute mean KL per trajectory on its action tokens.
        # Build an index map from flat action tokens back to traj ids.
        # We can reconstruct counts by iterating rows.
        traj_id_for_token = []
        counts = torch.zeros(B, dtype=torch.long, device=device)
        offset = 0
        for i in range(B):
            mrow = act_mask[i]
            n_i = int(mrow.sum().item())
            if n_i > 0:
                traj_id_for_token.extend([i] * n_i)
            counts[i] = n_i
            offset += n_i
        traj_id_for_token = torch.tensor(traj_id_for_token, dtype=torch.long, device=device)
        raw_rewards_t = torch.tensor(raw_rewards, dtype=torch.float, device=device)

        # Compute per-prompt group mean of shaped rewards
        group_mean = torch.zeros(B, dtype=torch.float, device=device)
        for pid in range(P):
            idxs = [i for i in range(B) if prompt_id_of[i] == pid]
            if not idxs:
                continue
            idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
            mean_val = raw_rewards_t[idxs_t].mean()
            group_mean[idxs_t] = mean_val

        # Advantage per trajectory, broadcast to its action tokens
        traj_adv = raw_rewards_t - group_mean  # (B,)

        # Build a flat tensor of advantages aligned with old_logp/new_logp on action tokens
        if kl_tok.numel() > 0:
            adv_flat = traj_adv[traj_id_for_token]
        else:
            adv_flat = torch.zeros(0, dtype=torch.float, device=device)

        # Normalize advantages (optional but usually helpful)
        if adv_flat.numel() > 1:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std().clamp_min(1e-6))

        # ----- UPDATE (policy-only PPO clipped objective) -----
        policy.train()
        logits_new, _, _ = policy(seq, None)  # ignore value head
        logp_full = torch.log_softmax(logits_new[:, :-1, :], dim=-1)
        labels = seq[:, 1:]
        new_logp_all = logp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        new_logp = new_logp_all[act_mask]

        # Mean KL over action tokens
        kl_now_ref_mean = (new_logp - ref_logp).mean() if new_logp.numel() > 0 else torch.tensor(0.0, device=device)

        out_loss = ppo_policy_only_losses(
            new_logp=new_logp,
            old_logp=old_logp,
            adv=adv_flat,
            clip_ratio=0.2,
            ent_coef=0.0,  # set >0 if you want entropy bonus from -new_logp mean
            kl_coef=args.kl_coef,
            kl_mean=kl_now_ref_mean,
        )
        loss = out_loss.total_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        policy.eval()

        # Some quick diagnostics (movement vs old, and now vs ref)
        with torch.no_grad():
            lp_post = model_logprobs(policy, seq)[act_mask]
            kl_move = (old_logp - lp_post).mean() if lp_post.numel() > 0 else torch.tensor(0.0, device=device)
            # KL(now || ref)
            kl_ref_now = (lp_post - ref_logp).mean() if lp_post.numel() > 0 else torch.tensor(0.0, device=device)

        step += 1
        if step % 10 == 0:
            print(
                f"step {step} | loss {loss.item():.4f}"
                f"| KL_move {kl_move.item():.6f} | KL_ref {kl_ref_now.item():.6f}"
            )

    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({'model': policy.state_dict(), 'config': {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
    }}, str(Path(args.out)/'model_last.pt'))
    print(f"Saved GRPO policy to {args.out}/model_last.pt")


if __name__ == '__main__':
    main()
