# SkyPilot DDP Training on Nebius

**Multi-node PyTorch DDP training on H100s in managed Kubernetes. It covers building and pushing a training container, provisioning GPU node groups on Nebius mk8s, and launching `torchrun` jobs through SkyPilot. We then measured how much the network actually costs when you scale from 1 to 4 GPUs.**

![PyTorch](https://img.shields.io/badge/PyTorch-DDP%20%2F%20NCCL-ee4c2c) ![SkyPilot](https://img.shields.io/badge/SkyPilot-k8s-blue) ![Kubernetes](https://img.shields.io/badge/Nebius-mk8s%20H100-black) ![Docker](https://img.shields.io/badge/Docker-NGC%20PyTorch-2496ed)

## Architecture

```
 Dockerfile (NGC PyTorch + HF stack) ──► Nebius Container Registry
                                              │
 SkyPilot API server ◄── sky launch ── skypilot/*.yaml
        │
        ▼
 Nebius mk8s cluster ── node group: N × 1×H100 (infra/*.json)
        │
        ▼
 torchrun --nnodes=$SKYPILOT_NUM_NODES --node_rank=$SKYPILOT_NODE_RANK \
          --master_addr=<first node IP> train.py      ← NCCL all-reduce across nodes
```

- **[`train.py`](train.py)** (experiment 1): causal-LM training on WikiText-2 with Hugging Face `Trainer` under DDP. It initializes NCCL from the torchrun environment variables, packs the tokens into fixed blocks, and prints a rank-0 banner with the NCCL and world-size setup. The model is set by the `MODEL_ID` environment variable.
- **[`train_gpt2_instrumented.py`](train_gpt2_instrumented.py)** (experiment 2): GPT-2 Large pretraining on WikiText-2. A DDP comm hook times every all-reduce and counts the bytes it moves, so communication overhead can be measured directly instead of guessed.
- **[`skypilot/`](skypilot/):** job specs. SkyPilot provides the node IPs and ranks, and the first node becomes the rendezvous master. NCCL debug logging is turned on so the transport selection shows up in the logs.
- **[`infra/`](infra/):** the mk8s node-group configs used (2 nodes and 4 nodes, 1× H100 each).

## Experiments

### 1. OPT-2.7B on 2 nodes
This established the end-to-end pipeline: the container, the cluster, SkyPilot and a 2-rank DDP run. Full log with NCCL initialization: [`runs/opt-2.7b-2node/`](runs/opt-2.7b-2node/).

### 2. GPT-2 Large (774M params): 1 GPU vs. 4 nodes × 1 GPU
The max per-device batch size, 16, was found by binary search over OOMs (256 ✗, 32 ✗, 8 ✓, 16 ✓).

| Metric | 1 GPU | 4 GPU DDP |
|---|---|---|
| Train runtime | **362 s** | 1,424 s |
| Train samples/s | 22.07 | 22.48 |
| Avg step time | **0.56 s** | 2.82 s |
| Eval samples/s | 58.4 | **215.5 (3.7×)** |
| Measured comm time | 0 s | 1,239 s |
| Avg comm per optimizer step | – | 2.48 s |
| Gradient payload / step (fp32) | 3.1 GB | 3.1 GB |

**Takeaways:**
- **DDP made training 4× slower.** About 90% of each step is gradient all-reduce: 3.1 GB per step crossing nodes.
- **The NCCL logs show why:** `Using network Socket`. The single-GPU nodes talk over plain TCP, not InfiniBand, so the interconnect sets the speed.
- **Evaluation scaled almost linearly (3.7×),** because it has no backward pass and no all-reduce.
- **The fix is topology, not code.** Use 8×H100 nodes with NVLink inside the node, or an InfiniBand fabric between nodes, before adding more ranks.

Full analysis: [`docs/report_ddp_scaling.pdf`](docs/report_ddp_scaling.pdf). Raw logs: [`runs/gpt2-large-1v4gpu/`](runs/gpt2-large-1v4gpu/).

## Run it

```bash
# 1. Build and push the image
docker build -t cr.<region>.nebius.cloud/<registry-id>/nebius-trainer:v1 .
docker push  cr.<region>.nebius.cloud/<registry-id>/nebius-trainer:v1

# 2. Create an mk8s cluster + GPU node group (see infra/), deploy the SkyPilot API server

# 3. Launch (edit infra: / image_id: in the YAML first)
sky launch -c ddp skypilot/train_job_opt-2.7b_2node.yaml
sky logs ddp
```

Assignment briefs: [multi-node setup](docs/assignment/hw_ddp_multinode_setup.md) (experiment 1) and [scaling analysis](docs/assignment/hw_ddp_scaling.md) (experiment 2).

## Context

This was an assignment for **Nebius Academy: AI Performance Engineering** (MLOps module: distributed training; Inference module: DDP scaling).

## Authors

**Nathanael Bloch** · **Tomer Porat**
