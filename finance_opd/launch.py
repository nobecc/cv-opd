"""Launch existing Slime/SGLang with this package's adapters; no scheduler side effects."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

from .config import ROOT, add_dependency_paths, read_config
from .protocol import assert_compatible_tokenizers, finite


def environment(config: dict) -> dict:
    env = dict(os.environ)
    paths = [str(ROOT), config["slime"], config["safactory"], str(Path(config["safactory"]) / "rl"),
             env.get("MEGATRON_HOME", "/root/Megatron-LM")]
    env.update(FINANCE_OPD_CONFIG=config["config_path"],
               PYTHONPATH=os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])),
               PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               CUDA_DEVICE_MAX_CONNECTIONS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False", PYTORCH_ALLOC_CONF="expandable_segments:False",
               TRANSFORMER_IMPL="local", WANDB_MODE="offline")
    env["PATH"] = "/usr/local/nvidia/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib64" + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    for key in ["NO_PROXY", "no_proxy"]:
        env[key] = ",".join(filter(None, [env.get(key), "localhost", "127.0.0.1"]))
    # Teachers live on cluster-internal pod IPs; never route them through an
    # inherited HTTP proxy (CIDR no_proxy entries are not honored by many
    # clients, so the exact host must be listed).
    teacher_url = config.get("teacher", {}).get("url")
    if isinstance(teacher_url, str) and teacher_url.startswith(("http://", "https://")):
        teacher_host = urlsplit(teacher_url).hostname
        if teacher_host:
            for key in ["NO_PROXY", "no_proxy"]:
                env[key] = ",".join(filter(None, [env[key], teacher_host]))
    return env


def model_arguments(config: dict) -> list[str]:
    script = Path(config["slime"]) / config["training"].get("model_args", "scripts/models/qwen3.5-9B.sh")
    # Argument vector + NUL framing; no eval or shell interpolation of config values.
    output = subprocess.check_output(
        ["bash", "-c", 'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"', "finance-opd", str(script)],
        env=environment(config))
    result = [item.decode() for item in output.split(b"\0") if item]
    if not result:
        raise ValueError(f"empty MODEL_ARGS from {script}")
    return result


def training_topology(train: dict) -> dict:
    """Validate total/per-node GPU geometry before allocating any cluster resources."""
    gpus = int(train.get("gpus", 8))
    nodes = int(train.get("nodes", 1))
    default_per_node = gpus if nodes == 1 else (gpus // nodes if nodes > 0 and gpus % nodes == 0 else 0)
    gpus_per_node = int(train.get("gpus_per_node", default_per_node))
    cp = int(train.get("context_parallel_size", gpus))
    tp = int(train.get("tensor_parallel_size", 1))
    pp = int(train.get("pipeline_parallel_size", 1))
    rollout_tp = int(train.get("rollout_tensor_parallel_size", 2))
    if min(gpus, nodes, gpus_per_node, cp, tp, pp, rollout_tp) < 1:
        raise ValueError("GPU counts, nodes, TP, PP, CP, and rollout TP must be positive")
    if gpus_per_node > 8:
        raise ValueError("each actor node supports at most 8 GPUs; increase nodes instead")
    if gpus != nodes * gpus_per_node:
        raise ValueError("training.gpus must equal nodes * gpus_per_node")
    if gpus % (tp * pp * cp):
        raise ValueError("GPU count must be a positive multiple of TP * PP * CP")
    if gpus % rollout_tp:
        raise ValueError("total GPU count must be divisible by rollout tensor parallel size")
    return {"gpus": gpus, "nodes": nodes, "gpus_per_node": gpus_per_node,
            "tp": tp, "pp": pp, "cp": cp, "rollout_tp": rollout_tp,
            "data_parallel_size": gpus // (tp * pp * cp),
            "rollout_engines": gpus // rollout_tp}


def training_arguments(config: dict) -> list[str]:
    train, runtime = config["training"], config["runtime"]
    opd_coef = finite(train.get("opd_coef", 1.0), "opd coefficient")
    if opd_coef <= 0:
        raise ValueError("opd coefficient must be positive")
    topology = training_topology(train)
    gpus, nodes = topology["gpus"], topology["nodes"]
    gpus_per_node = topology["gpus_per_node"]
    cp, tp, pp = topology["cp"], topology["tp"], topology["pp"]
    qkv_format = train.get("qkv_format", "thd")
    if qkv_format not in {"thd", "bshd"}:
        raise ValueError("qkv_format must be thd or bshd")
    checkpoint_format = train.get("checkpoint_format", "torch_dist")
    if checkpoint_format not in {"torch", "torch_dist"}:
        raise ValueError("checkpoint_format must be torch or torch_dist")
    offload_fraction = float(train.get("optimizer_offload_fraction", 1.0))
    if not math.isfinite(offload_fraction) or not 0 < offload_fraction <= 1:
        raise ValueError("optimizer_offload_fraction must be in (0, 1]")
    offload_train = train.get("offload_train", True)
    if not isinstance(offload_train, bool):
        raise ValueError("offload_train must be a boolean")
    weight_decay = float(train.get("weight_decay", 0.1))
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    batch = int(train.get("rollout_batch_size", 1))
    samples = int(train.get("samples_per_task", 1))
    if batch < 1 or samples < 1:
        raise ValueError("rollout_batch_size and samples_per_task must be positive")
    if batch * samples % topology["data_parallel_size"]:
        raise ValueError("episode groups per update must be divisible by actor data parallel size")
    length = runtime.get("context_length", 262144)
    checkpoint = str(Path(config["student_model"]).resolve())
    args = model_arguments(config) + [
        "--actor-num-nodes", str(nodes), "--actor-num-gpus-per-node", str(gpus_per_node),
        "--rollout-num-gpus", str(gpus), "--num-gpus-per-node", str(gpus_per_node), "--colocate",
        "--train-backend", "megatron", "--megatron-to-hf-mode", "bridge", "--transformer-impl", "local",
        "--custom-model-provider-path", "rl.examples.pi_finance.model_provider.model_provider",
        "--hf-checkpoint", checkpoint, "--load", train.get("load_checkpoint", checkpoint),
        "--ckpt-format", checkpoint_format,
        "--save", str(Path(config["output"]) / "checkpoints"), "--save-interval", str(train.get("save_interval", 5)),
        "--prompt-data", config["dataset"], "--input-key", "task", "--metadata-key", "metadata", "--rollout-shuffle",
        # Request Slime's conversation wrapper even for text-only tasks. Qwen
        # checkpoints load a processor which requires a message list. Pi still
        # owns chat templating; the rollout hook unwraps the original task text.
        "--multimodal-keys", "{}",
        "--custom-generate-function-path", "finance_opd.rollout.generate",
        "--custom-reward-post-process-path", "finance_opd.advantages.post_process_weights",
        "--custom-advantage-function-path", "finance_opd.advantages.compute_advantages",
        "--reward-key", "opd_weight", "--disable-rewards-normalization", "--use-rollout-logprobs",
        "--advantage-estimator", "grpo", "--opd-kl-coef", str(opd_coef),
        "--num-rollout", str(train.get("updates", 20)), "--rollout-batch-size", str(batch),
        "--n-samples-per-prompt", str(samples), "--global-batch-size", str(batch * samples),
        "--num-steps-per-rollout", "1", "--rollout-temperature", "1", "--rollout-top-p", "1",
        "--rollout-max-response-len", str(length), "--rollout-max-context-len", str(length),
        "--tensor-model-parallel-size", str(tp), "--pipeline-model-parallel-size", str(pp),
        "--context-parallel-size", str(cp), "--cp-comm-type", "p2p",
        # Dense model: expert TP must match tensor TP, otherwise MCore's
        # legacy checkpoint gate treats TP ranks 1..N as expert-DP replicas.
        "--expert-model-parallel-size", "1", "--expert-tensor-parallel-size", str(tp),
        "--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", "1",
        "--attention-dropout", "0", "--hidden-dropout", "0", "--attention-backend", "flash",
        "--accumulate-allreduce-grads-in-fp32", "--attention-softmax-in-fp32",
        "--entropy-coef", "0", "--eps-clip", "0.2", "--eps-clip-high", "0.2",
        "--optimizer", "adam", "--lr", str(train.get("learning_rate", 5e-7)),
        "--lr-decay-style", "constant", "--weight-decay", str(weight_decay), "--adam-beta1", "0.9", "--adam-beta2", "0.98",
        "--optimizer-cpu-offload", "--optimizer-offload-fraction", str(offload_fraction),
        "--overlap-cpu-optimizer-d2h-h2d", "--use-precision-aware-optimizer",
        "--rollout-num-gpus-per-engine", str(topology["rollout_tp"]),
        "--sglang-mem-fraction-static", str(train.get("rollout_memory_fraction", 0.65)),
        "--sglang-tool-call-parser", "qwen3_coder", "--sglang-reasoning-parser", "qwen3",
        "--sglang-context-length", str(length), "--sglang-attention-backend", "fa3",
        "--sglang-max-running-requests", str(train.get("max_running_requests", 4)),
        "--sglang-chunked-prefill-size", "16384", "--sglang-mamba-scheduler-strategy", "extra_buffer",
        "--sglang-disable-cuda-graph", "--sglang-log-level", "warning", "--sglang-log-level-http", "error",
        "--save-debug-rollout-data", str(Path(config["output"]) / "rollout/slime-{rollout_id}.pt"),
    ]
    if config.get("continuation", {}).get("enabled", False):
        args += ["--dynamic-sampling-filter-path", "finance_opd.training_filter.require_cv_signal"]
    save_optimizer = train.get("save_optimizer", True)
    if not isinstance(save_optimizer, bool):
        raise ValueError("save_optimizer must be a boolean")
    if not save_optimizer:
        args.append("--no-save-optim")
    chunk_size = train.get("log_probs_chunk_size")
    if chunk_size is not None:
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("log_probs_chunk_size must be a positive integer")
        args += ["--log-probs-chunk-size", str(chunk_size)]
    recompute_loss = train.get("recompute_loss_function", False)
    if not isinstance(recompute_loss, bool):
        raise ValueError("recompute_loss_function must be a boolean")
    if recompute_loss:
        args.append("--recompute-loss-function")
    reset_optimizer = train.get("reset_optimizer", False)
    if not isinstance(reset_optimizer, bool):
        raise ValueError("reset_optimizer must be a boolean")
    if reset_optimizer:
        if not train.get("load_checkpoint"):
            raise ValueError("reset_optimizer requires a saved model checkpoint")
        args += ["--no-load-optim", "--no-load-rng"]
    if not offload_train:
        # Keep GPU-resident optimizer states on GPU between rollouts. Moving
        # them back to host defeats partial optimizer offload's memory benefit.
        args += ["--no-offload-train", "--offload-rollout"]
    args += ["--qkv-format", qkv_format]
    if qkv_format == "bshd":
        # Megatron GDN in the deployed image rejects packed_seq_params.
        # Slime's bshd path passes None and requires static micro batches.
        args += ["--micro-batch-size", "1"]
    else:
        args += ["--use-dynamic-batch-size", "--max-tokens-per-gpu",
                 str(train.get("max_tokens_per_gpu", length // cp))]
    if tp > 1:
        args.append("--sequence-parallel")
    return args


def teacher_arguments(config: dict) -> list[str]:
    from .teacher_transport import transport_name
    if transport_name(config) != "sglang":
        raise ValueError("the local teacher launcher supports only the sglang transport")
    teacher = config["teacher"]
    parsed = urlsplit(teacher["url"])
    if parsed.scheme != "http" or not parsed.port or parsed.path not in {"", "/"}:
        raise ValueError("local teacher launcher expects http://host:port without a path")
    return [sys.executable, "-m", "sglang.launch_server", "--model-path", teacher["model"],
            "--host", teacher.get("listen_host", "0.0.0.0"), "--port", str(parsed.port),
            "--tp", str(teacher.get("tensor_parallel_size", 2)),
            "--context-length", str(teacher.get("context_length", 262144)),
            "--chunked-prefill-size", str(teacher.get("chunked_prefill_size", 16384)),
            "--mem-fraction-static", str(teacher.get("memory_fraction", 0.8)),
            *teacher.get("extra_args", [])]


def doctor(config: dict, *, offline=False) -> dict:
    add_dependency_paths(config)
    checks = {}
    if config.get("continuation", {}).get("enabled", False):
        try:
            from .value import validate_dataset_references
            validate_dataset_references(config)
            checks["continuation_supervision"] = True
        except (ValueError, OSError, ImportError) as error:
            checks["continuation_supervision"] = False
            checks["continuation_error"] = str(error)
    for key, relative in [("slime", "train.py"), ("pi_finance", "runtime.ts"),
                          ("safactory", "rl/examples/pi_finance/mask_builder.py")]:
        checks[key] = (Path(config[key]) / relative).is_file()
    checks["dataset"] = Path(config["dataset"]).is_file()
    try:
        from rl.examples.pi_finance.qwen_protocol import validate_native_parser
        checks["native_qwen_parser"] = validate_native_parser()
    except (ImportError, ValueError, AttributeError) as error:
        checks["native_qwen_parser"] = False
        checks["native_qwen_parser_error"] = str(error)
    for name in ["torch", "ray", "sglang", "aiohttp", "yaml", "transformers", "qwen_vl_utils"]:
        checks[f"import_{name}"] = importlib.util.find_spec(name) is not None
    try:
        from .teacher_transport import validate_teacher_config
        validate_teacher_config(config)
        checks["teacher_transport"] = True
    except ValueError as error:
        checks["teacher_transport"] = False
        checks["teacher_transport_error"] = str(error)
    # In vllm mode teacher.model is a served name; tokenizer compatibility is
    # checked against the local snapshot referenced by tokenizer_path.
    teacher_tokenizer = config["teacher"].get("tokenizer_path") or config["teacher"]["model"]
    try:
        assert_compatible_tokenizers(config["student_model"], teacher_tokenizer)
        checks["tokenizers_compatible"] = True
    except (ValueError, OSError) as error:
        checks["tokenizers_compatible"] = False
        checks["tokenizer_error"] = str(error)
    env = environment(config)
    python = config["runtime"].get("pi_python", sys.executable)
    native_env = dict(env, PYTHONPATH=config["pi_finance"])
    try:
        probe = subprocess.run([python, "-m", "pifinance.run", "--help"], cwd=config["pi_finance"],
                               env=native_env, capture_output=True, timeout=30)
        checks["native_python"] = probe.returncode == 0
        node_probe = subprocess.run([python, "-c", "from pifinance.run import resolve_node; print(resolve_node())"],
                                    cwd=config["pi_finance"], env=native_env, capture_output=True, timeout=30)
        checks["native_node"] = node_probe.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        checks["native_python"] = checks["native_node"] = False
    if not offline:
        import urllib.request
        from .teacher_transport import parse_vllm_models, transport_name
        try:
            url = config["teacher"]["url"].rstrip("/")
            key = os.environ.get("FINANCE_OPD_TEACHER_API_KEY")
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            if transport_name(config) == "vllm":
                request = urllib.request.Request(url + "/v1/models", headers=headers)
                with opener.open(request, timeout=10) as response:
                    served = parse_vllm_models(json.load(response))
                checks["teacher_model_matches"] = served == config["teacher"]["model"]
            else:
                request = urllib.request.Request(url + "/get_model_info", headers=headers)
                with opener.open(request, timeout=10) as response:
                    info = json.load(response)
                checks["teacher_model_matches"] = Path(info["model_path"]).resolve() == Path(config["teacher"]["model"]).resolve()
        except Exception:
            checks["teacher_model_matches"] = False
    passed = all(v is True for k, v in checks.items() if not k.endswith("_error"))
    return {"checks": checks, "checks_passed": passed, "training_ready": passed and not offline,
            "teacher_checked": not offline, "gpu_training_executed": False}


def propagated_environment(config: dict, env: dict) -> dict:
    """Environment required by custom rollout code and distributed Ray actors."""
    propagated = {key: env[key] for key in ["FINANCE_OPD_CONFIG", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
                  "TOKENIZERS_PARALLELISM", "PYTHONUNBUFFERED", "CUDA_DEVICE_MAX_CONNECTIONS",
                  "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
                  "TRANSFORMER_IMPL", "WANDB_MODE", "NO_PROXY", "no_proxy", "PATH", "LD_LIBRARY_PATH"]}
    for key in ["PI_NODE", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                "NODE_USE_ENV_PROXY", "DATAELF_FINANCE_USER_AGENT", "MASTER_ADDR", "MASTER_PORT",
                "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "NCCL_IB_DISABLE",
                "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME"]:
        if key in env:
            propagated[key] = env[key]
    if "FINANCE_OPD_TEACHER_API_KEY" in env:
        propagated["FINANCE_OPD_TEACHER_API_KEY"] = env["FINANCE_OPD_TEACHER_API_KEY"]
    if config.get("continuation", {}).get("enabled", False):
        for key, value in env.items():
            if key.startswith(("EVAL_FINANCE_", "PI_FINANCE_EVAL_", "DDR_REWARD_JUDGE_")):
                propagated[key] = value
    return propagated


def prepare_training(config: dict) -> tuple[list[str], dict]:
    """Run rank-zero preflight and create the one shared training output."""
    hold = Path(config["output"]).parent / "configs" / "TRAINING_HOLD.json"
    if hold.is_file():
        raise RuntimeError(f"Owned run is screening-only pending a controlled admission-method handoff; see {hold}")
    command = [sys.executable, str(Path(config["slime"]) / "train.py"), *training_arguments(config)]
    report = doctor(config)
    if not report["training_ready"]:
        raise RuntimeError("training preflight failed: " + json.dumps(report))
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "rollout").mkdir()
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    env = environment(config)
    env["FINANCE_OPD_CONFIG"] = str(out / "run_config.json")
    return command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "teacher", "doctor"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen35_9b.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline", action="store_true", help="doctor only: omit the teacher HTTP probe")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.mode == "doctor":
        report = doctor(config, offline=args.offline)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["checks_passed"] else 1)
    if args.mode == "teacher":
        command = teacher_arguments(config)
        if args.dry_run:
            print(shlex.join(command))
            return
        os.execvpe(command[0], command, environment(config))
    if args.dry_run:
        print(shlex.join([sys.executable, str(Path(config["slime"]) / "train.py"),
                          *training_arguments(config)]))
        return
    command, env = prepare_training(config)
    os.environ.update(env)
    os.chdir(ROOT)
    add_dependency_paths(config)
    import ray
    ray.init(address=config["training"].get("ray_address"),
             runtime_env={"env_vars": propagated_environment(config, env)})
    spec = importlib.util.spec_from_file_location("finance_opd._slime_entry", Path(config["slime"]) / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = command[1:]
    module.train(module.parse_args())


if __name__ == "__main__":
    main()
