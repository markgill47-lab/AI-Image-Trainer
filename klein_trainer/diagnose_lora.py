#!/usr/bin/env python3
"""
LoRA Training Diagnostic
========================
Runs the first few training steps with heavy instrumentation to answer:
"Are LoRA weights actually being updated during training?"

Usage:
    python -m klein_trainer.diagnose_lora --gui-config gui_config.json
    python -m klein_trainer.diagnose_lora --dataset /path/to/data --vae /path/to/vae.safetensors --transformer /path/to/klein.safetensors

Tests for:
    1. LoRA layers injected at expected target modules
    2. LoRA weights are in bf16/fp16/fp32 (NOT int8)
    3. LoRA parameters have requires_grad=True
    4. LoRA parameters are in the optimizer
    5. Gradients flow to LoRA parameters (grad_norm > 0 after backward)
    6. LoRA weights actually change between steps (weights DIFFER after optimizer.step)
    7. Quantization hasn't cascade-frozen the LoRA layers
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

# Ensure we can import klein_trainer when run as a script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from klein_trainer.config import KleinConfig
from klein_trainer.trainer import KleinTrainer


# ─────────────────────────────────────────────────────────────────────
# ANSI colors for verdict messages
# ─────────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}!{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}\n" + "─" * len(msg))


# ─────────────────────────────────────────────────────────────────────
# Diagnostic checks
# ─────────────────────────────────────────────────────────────────────

def check_lora_injection(lora_network):
    """Check which target modules got LoRA layers."""
    header("1. LoRA Injection Check")

    total = len(lora_network.lora_layers)
    if total == 0:
        fail(f"ZERO LoRA layers injected! Target modules: {lora_network.target_modules}")
        fail("This means the base model's module names don't match target_modules.")
        print("\nList of all Linear layers in the transformer (first 30):")
        i = 0
        for name, module in lora_network.transformer.named_modules():
            if isinstance(module, nn.Linear):
                print(f"    {name}")
                i += 1
                if i >= 30:
                    print(f"    ... ({sum(1 for n,m in lora_network.transformer.named_modules() if isinstance(m, nn.Linear))} total)")
                    break
        return False

    ok(f"{total} LoRA layers injected")

    # Breakdown by target pattern
    by_target = defaultdict(int)
    for name in lora_network.lora_layers:
        for target in lora_network.target_modules:
            if target in name:
                by_target[target] += 1
                break
    for target, count in sorted(by_target.items(), key=lambda x: -x[1]):
        print(f"    {target:20s} → {count} layers")

    trainable = lora_network.get_trainable_params()
    ok(f"Trainable params: {trainable:,}")
    return True


def check_lora_dtypes(lora_network):
    """Check LoRA weights are in a trainable dtype, not int8."""
    header("2. LoRA Dtype Check")

    # Sample first layer
    first_name = next(iter(lora_network.lora_layers))
    first_layer = lora_network.lora_layers[first_name]

    a_dtype = first_layer.lora_A.weight.dtype
    b_dtype = first_layer.lora_B.weight.dtype

    print(f"    First layer: {first_name}")
    print(f"    lora_A.weight.dtype: {a_dtype}")
    print(f"    lora_B.weight.dtype: {b_dtype}")

    trainable_dtypes = {torch.float32, torch.bfloat16, torch.float16}
    all_good = True
    for name, layer in lora_network.lora_layers.items():
        if layer.lora_A.weight.dtype not in trainable_dtypes:
            fail(f"{name}.lora_A has dtype {layer.lora_A.weight.dtype} — CANNOT TRAIN")
            all_good = False
        if layer.lora_B.weight.dtype not in trainable_dtypes:
            fail(f"{name}.lora_B has dtype {layer.lora_B.weight.dtype} — CANNOT TRAIN")
            all_good = False

    if all_good:
        ok("All LoRA weights are in a trainable dtype")
    return all_good


def check_requires_grad(lora_network):
    """Check LoRA parameters have requires_grad=True."""
    header("3. requires_grad Check")

    not_requiring_grad = []
    total = 0
    for name, layer in lora_network.lora_layers.items():
        for sub, p in [("lora_A", layer.lora_A.weight), ("lora_B", layer.lora_B.weight)]:
            total += 1
            if not p.requires_grad:
                not_requiring_grad.append(f"{name}.{sub}")

    if not_requiring_grad:
        fail(f"{len(not_requiring_grad)} of {total} LoRA params do NOT have requires_grad=True")
        for n in not_requiring_grad[:5]:
            print(f"    {n}")
        if len(not_requiring_grad) > 5:
            print(f"    ... and {len(not_requiring_grad) - 5} more")
        fail("Optimizer cannot update these — this IS your flat loss cause.")
        return False

    ok(f"All {total} LoRA parameters have requires_grad=True")
    return True


def check_optimizer_params(optimizer, lora_network):
    """Verify the optimizer actually contains the LoRA parameters."""
    header("4. Optimizer Param Check")

    opt_params = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            opt_params.add(id(p))

    lora_param_ids = set()
    for layer in lora_network.lora_layers.values():
        lora_param_ids.add(id(layer.lora_A.weight))
        lora_param_ids.add(id(layer.lora_B.weight))

    missing = lora_param_ids - opt_params
    if missing:
        fail(f"{len(missing)} LoRA parameters are NOT in the optimizer")
        fail("Optimizer.step() cannot update parameters it doesn't know about.")
        return False

    extra = opt_params - lora_param_ids
    ok(f"All {len(lora_param_ids)} LoRA params are in the optimizer")
    if extra:
        warn(f"{len(extra)} extra non-LoRA params in optimizer (may be normal)")
    return True


def snapshot_lora_weights(lora_network):
    """Return {name: {'A': norm, 'B': norm}} dict of L2 norms."""
    snap = {}
    with torch.no_grad():
        for name, layer in lora_network.lora_layers.items():
            snap[name] = {
                "A": layer.lora_A.weight.detach().float().norm().item(),
                "B": layer.lora_B.weight.detach().float().norm().item(),
            }
    return snap


def snapshot_gradient_norms(lora_network):
    """Return {name: {'A': grad_norm, 'B': grad_norm}}."""
    snap = {}
    with torch.no_grad():
        for name, layer in lora_network.lora_layers.items():
            ga = layer.lora_A.weight.grad
            gb = layer.lora_B.weight.grad
            snap[name] = {
                "A": ga.float().norm().item() if ga is not None else None,
                "B": gb.float().norm().item() if gb is not None else None,
            }
    return snap


def weight_norms_summary(snap):
    a_norms = [v["A"] for v in snap.values()]
    b_norms = [v["B"] for v in snap.values()]
    return {
        "A_mean": sum(a_norms) / len(a_norms),
        "A_max":  max(a_norms),
        "B_mean": sum(b_norms) / len(b_norms),
        "B_max":  max(b_norms),
    }


def weight_change_summary(before, after):
    changes_A, changes_B = [], []
    for name in before:
        changes_A.append(abs(after[name]["A"] - before[name]["A"]))
        changes_B.append(abs(after[name]["B"] - before[name]["B"]))
    return {
        "A_total_change": sum(changes_A),
        "A_mean_change":  sum(changes_A) / len(changes_A),
        "A_max_change":   max(changes_A),
        "B_total_change": sum(changes_B),
        "B_mean_change":  sum(changes_B) / len(changes_B),
        "B_max_change":   max(changes_B),
        "layers_with_A_changes": sum(1 for c in changes_A if c > 1e-9),
        "layers_with_B_changes": sum(1 for c in changes_B if c > 1e-9),
        "total_layers": len(changes_A),
    }


def run_training_steps(trainer, num_steps=5):
    """Run N training steps with instrumentation, return per-step diagnostic data."""
    header(f"5–7. Running {num_steps} Instrumented Training Steps")

    lora_network = trainer.lora_network
    optimizer = trainer.optimizer

    # Set to training mode (critical!)
    lora_network.train()

    # Get the dataloader
    from klein_trainer.dataset import create_dataloader
    dataloader = create_dataloader(
        dataset=trainer.dataset,
        batch_size=trainer.config.batch_size,
        num_workers=0,
    )
    data_iter = iter(dataloader)

    initial_snap = snapshot_lora_weights(lora_network)
    print(f"\n  Initial LoRA weight L2 norms:")
    init_summary = weight_norms_summary(initial_snap)
    print(f"    lora_A mean norm: {init_summary['A_mean']:.4f}  (should be > 0: Kaiming init)")
    print(f"    lora_B mean norm: {init_summary['B_mean']:.4e} (should be ~0: zero init)")

    prev_snap = initial_snap
    per_step_data = []

    for step in range(1, num_steps + 1):
        step_start = time.time()
        print(f"\n  ── Step {step} ──")

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Reset gradients
        optimizer.zero_grad()

        # Forward + backward via the trainer's own train_step method
        try:
            loss, _per_sample = trainer._training_step(batch)
        except Exception as e:
            fail(f"train_step raised: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Capture gradients BEFORE optimizer step
        grad_snap = snapshot_gradient_norms(lora_network)
        grad_A_norms = [g["A"] for g in grad_snap.values() if g["A"] is not None]
        grad_B_norms = [g["B"] for g in grad_snap.values() if g["B"] is not None]
        grad_A_none  = sum(1 for g in grad_snap.values() if g["A"] is None)
        grad_B_none  = sum(1 for g in grad_snap.values() if g["B"] is None)

        print(f"    Loss: {loss:.4f}")
        print(f"    lora_A grad norm:  mean={sum(grad_A_norms)/max(len(grad_A_norms),1):.4e}  max={max(grad_A_norms) if grad_A_norms else 0:.4e}  (None: {grad_A_none})")
        print(f"    lora_B grad norm:  mean={sum(grad_B_norms)/max(len(grad_B_norms),1):.4e}  max={max(grad_B_norms) if grad_B_norms else 0:.4e}  (None: {grad_B_none})")

        # Optimizer step
        optimizer.step()

        # Snapshot weights after step
        post_snap = snapshot_lora_weights(lora_network)

        # Compare to previous step
        delta = weight_change_summary(prev_snap, post_snap)
        print(f"    lora_A weight change: total={delta['A_total_change']:.4e}  max={delta['A_max_change']:.4e}  ({delta['layers_with_A_changes']}/{delta['total_layers']} layers changed)")
        print(f"    lora_B weight change: total={delta['B_total_change']:.4e}  max={delta['B_max_change']:.4e}  ({delta['layers_with_B_changes']}/{delta['total_layers']} layers changed)")
        print(f"    (step took {time.time()-step_start:.1f}s)")

        per_step_data.append({
            "step": step,
            "loss": loss,
            "grad_A_mean": sum(grad_A_norms)/max(len(grad_A_norms),1),
            "grad_B_mean": sum(grad_B_norms)/max(len(grad_B_norms),1),
            "grad_A_none": grad_A_none,
            "grad_B_none": grad_B_none,
            "A_total_change": delta["A_total_change"],
            "B_total_change": delta["B_total_change"],
            "layers_with_A_changes": delta["layers_with_A_changes"],
            "layers_with_B_changes": delta["layers_with_B_changes"],
            "total_layers": delta["total_layers"],
        })

        prev_snap = post_snap

    # Final summary vs initial
    final_delta = weight_change_summary(initial_snap, prev_snap)
    print(f"\n  Cumulative change over {num_steps} steps (vs initial):")
    print(f"    lora_A total change: {final_delta['A_total_change']:.4e}")
    print(f"    lora_B total change: {final_delta['B_total_change']:.4e}")
    print(f"    Layers with B changes: {final_delta['layers_with_B_changes']}/{final_delta['total_layers']}")

    return {
        "per_step": per_step_data,
        "cumulative": final_delta,
    }


def render_verdict(results):
    header("VERDICT")

    if results is None:
        fail("Could not complete diagnostic — training step raised an exception.")
        return 1

    per_step = results["per_step"]
    cumulative = results["cumulative"]

    issues = []

    # Check 1: Any gradients at all?
    any_grad_A = any(s["grad_A_mean"] > 0 for s in per_step)
    any_grad_B = any(s["grad_B_mean"] > 0 for s in per_step)

    if not any_grad_A and not any_grad_B:
        issues.append("No gradients on ANY LoRA parameter — backward pass isn't touching them.")

    # Check 2: Gradients on B specifically (B starts at zero, needs gradients to learn)
    if not any_grad_B:
        issues.append("lora_B has no gradients — since B is initialized to zero, this means LoRA contributes nothing to the forward pass.")

    # Check 3: None gradients?
    any_none_grads = any(s["grad_A_none"] > 0 or s["grad_B_none"] > 0 for s in per_step)
    if any_none_grads:
        issues.append("Some LoRA parameters have grad=None — they weren't reached by backward(). Likely cause: LoRA wasn't called in forward pass, OR gradient checkpointing dropped it.")

    # Check 4: Did weights actually change?
    if cumulative["layers_with_B_changes"] == 0:
        issues.append("ZERO layers had lora_B weight changes over all steps — optimizer.step() had no effect.")
    elif cumulative["layers_with_B_changes"] < cumulative["total_layers"] // 2:
        issues.append(f"Only {cumulative['layers_with_B_changes']}/{cumulative['total_layers']} LoRA layers had weight changes. The rest are not training.")

    # Check 5: Loss variance
    losses = [s["loss"] for s in per_step]
    loss_range = max(losses) - min(losses)
    if loss_range < 0.01:
        issues.append(f"Loss varied by only {loss_range:.4f} across {len(losses)} steps. In flow matching with random timesteps, natural variance should be > 0.05 even with no learning.")

    if issues:
        fail(f"{len(issues)} issue(s) detected:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        print()
        print(f"{BOLD}This is why loss is flatlining. Next step: inspect the specific hypothesis above.{RESET}")
        return 1

    ok(f"LoRA training appears healthy:")
    print(f"    - Gradients flow to LoRA parameters (mean B grad: {per_step[-1]['grad_B_mean']:.4e})")
    print(f"    - Weights are updating ({cumulative['layers_with_B_changes']}/{cumulative['total_layers']} lora_B layers changed)")
    print(f"    - Natural loss variance across steps: {loss_range:.4f}")
    print()
    print(f"{BOLD}If loss still flatlines over 100+ steps, the trainer is working but learning the wrong thing.{RESET}")
    print("Likely causes: learning rate, LoRA scale, dataset/caption issues, or quantization precision loss.")
    return 0


# ─────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────

def build_config_from_gui(gui_config_path):
    """Build KleinConfig from the GUI's gui_config.json."""
    with open(gui_config_path) as f:
        gui = json.load(f)

    paths = gui.get("paths", {})
    model = gui.get("model", {})
    training = gui.get("training", {})

    is_4b = "4b" in model.get("selected", "").lower()
    variant = "4B" if is_4b else "9B"

    return KleinConfig(
        model_variant=variant,
        model_path=paths.get("transformer_path") or None,
        vae_path=paths.get("vae_path") or None,
        text_encoder_repo=paths.get("text_encoder_repo") or None,
        dataset_path=paths.get("dataset_dir", ""),
        output_dir=paths.get("output_dir", "./output"),
        output_name="_diagnose",
        max_train_steps=10,
        learning_rate=float(training.get("learning_rate", 1e-4)),
        lora_rank=int(training.get("lora_rank", 32)),
        lora_alpha=int(training.get("lora_alpha", 32)),
        sample_every_n_steps=9999,  # skip sampling
        save_every_n_steps=9999,    # skip saving
        log_every_n_steps=1,
    )


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRA training diagnostic")
    parser.add_argument("--gui-config", help="Path to gui_config.json (convenient for using current GUI settings)")
    parser.add_argument("--dataset", help="Dataset directory")
    parser.add_argument("--vae", help="VAE .safetensors path")
    parser.add_argument("--transformer", help="Transformer .safetensors path")
    parser.add_argument("--text-encoder-repo", default=None, help="HuggingFace repo for text encoder (default: Klein repo)")
    parser.add_argument("--variant", default="9B", choices=["4B", "9B"])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=5, help="Number of diagnostic training steps to run")
    parser.add_argument("--no-quantize", action="store_true", help="Disable int8 quantization (to test if quanto is the cause)")
    args = parser.parse_args()

    # Build config
    if args.gui_config:
        config = build_config_from_gui(args.gui_config)
    else:
        if not (args.dataset and args.vae):
            parser.error("Need --gui-config OR --dataset AND --vae")
        config = KleinConfig(
            model_variant=args.variant,
            model_path=args.transformer,
            vae_path=args.vae,
            text_encoder_repo=args.text_encoder_repo,
            dataset_path=args.dataset,
            output_dir="./output",
            output_name="_diagnose",
            max_train_steps=args.steps + 1,
            learning_rate=args.lr,
            lora_rank=args.rank,
            lora_alpha=args.alpha,
            sample_every_n_steps=9999,
            save_every_n_steps=9999,
            log_every_n_steps=1,
        )

    if args.no_quantize:
        config.quantize_transformer = False
        config.quantize_text_encoder = False
        print(f"\n{YELLOW}NOTE: Quantization disabled. This needs ~18GB VRAM for 9B.{RESET}")

    print(f"{BOLD}Klein LoRA Training Diagnostic{RESET}")
    print(f"  Variant:         {config.model_variant}")
    print(f"  Dataset:         {config.dataset_path}")
    print(f"  LR:              {config.learning_rate}")
    print(f"  LoRA rank/alpha: {config.lora_rank}/{config.lora_alpha}")
    print(f"  Quantize:        {config.quantize_transformer}")

    # Set up the trainer (loads VAE, transformer, LoRA, text encoder, dataset, optimizer)
    header("Loading trainer (this takes ~1 minute)...")
    trainer = KleinTrainer(config)
    try:
        trainer.setup()
    except Exception as e:
        fail(f"Trainer setup failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 2

    # Run checks
    all_checks_passed = True
    all_checks_passed &= check_lora_injection(trainer.lora_network)
    all_checks_passed &= check_lora_dtypes(trainer.lora_network)
    all_checks_passed &= check_requires_grad(trainer.lora_network)
    all_checks_passed &= check_optimizer_params(trainer.optimizer, trainer.lora_network)

    if not all_checks_passed:
        header("VERDICT")
        fail("Static checks failed — fix these first before running training steps.")
        return 1

    # Run actual training steps
    results = run_training_steps(trainer, num_steps=args.steps)
    return render_verdict(results)


if __name__ == "__main__":
    sys.exit(main())
