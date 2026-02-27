# Research Note: Rollback Netcode, Delay Frames, and World Model Prediction

## The Parallel

Rollback netcode and autoregressive world models face the same fundamental problem: predicting future game state from incomplete information. Both break on the same class of events — instant-onset transitions (frame-1 moves, one-frame state changes like TURNING).

## How Slippi Handles It

From the [Slippi FAQ](https://github.com/project-slippi/slippi-launcher/blob/main/FAQ.md):

> Delay frames are how we account for the time it takes to send an opponent your input. Since we have removed about 1.5 frames of visual delay from Melee, adding 2 frames brings us very close to a CRT experience. Using a 120hz+ monitor removes an additional half frame to bring us in line with CRT.
>
> A single delay frame is equal to 4 buffer in traditional Dolphin netplay. Our recommended default is 2 frame delay (8 buffer). We suggest using 2 frame delay for connections up to 130ms ping.

Slippi's approach:
1. **Accept that frame-1 predictions will be wrong** — don't try to predict instant-onset moves
2. **Add delay frames** — 2 frames of input delay means you always have 2 frames of "future" controller data before you need to render
3. **Rollback when wrong** — if the prediction was wrong, re-simulate from the last known-good state

## What This Means for Us

Our world model's worst predictions are exactly the cases rollback netcode handles with delay frames:

| Our failure mode | Rollback equivalent |
|-----------------|---------------------|
| TURNING (1-frame, 28% change_acc) | Frame-1 move prediction — rollback accepts these will be wrong |
| Aerial attack onset (BAIR 21%, FAIR 13%) | Attack startup on reaction — delay frames give advance notice |
| Damage transitions (24% change_acc) | Getting hit — inherently unpredictable without seeing opponent's exact timing |

### Delay frame analogy for training

Our `lookahead` parameter in EncodingConfig is conceptually a delay frame. With `lookahead=0`, the model predicts frame t's state given controller input at frame t — zero delay. With `lookahead=1`, it would predict frame t+1 given ctrl(t) and ctrl(t+1) — one frame of "future" information.

We tested 1-frame lookahead (Exp 3a) and got +6.3pp change_acc but traded position accuracy. The tradeoff makes sense: with more future information, action prediction improves (you know what buttons are coming), but physics prediction gets harder (predicting 2 frames of physics instead of 1).

### Design implications

Rather than trying to eliminate frame-1 prediction errors, we might:

1. **Accept the TURNING/damage ceiling.** These transitions are inherently hard to predict one frame in advance. Rollback netcode doesn't solve them — it copes with them.

2. **Design the autoregressive loop to be rollback-tolerant.** When the model mispredicts an action transition, have a mechanism to "snap" back to a plausible state rather than letting the error compound. This is closer to Slippi's rollback than to our current "feed errors forward" approach.

3. **Use delay frames for policy play.** When two policies play inside the world model, give each policy 1-2 frames of the opponent's future controller input (matching Slippi's default 2-frame delay). This is more realistic to actual Melee netplay AND easier for the world model to predict.

## No Action Item Yet

This is a research note. The connection between rollback netcode and autoregressive world models is worth keeping in mind as we design experiments — particularly E004 (scheduled sampling) and any future work on the rollout harness.
