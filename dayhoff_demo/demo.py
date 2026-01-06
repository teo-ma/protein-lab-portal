import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dayhoff-3b-GR-HM-c demo: sample protein generation (Transformers)."
    )
    parser.add_argument(
        "--model",
        default="microsoft/Dayhoff-3b-GR-HM-c",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Max generated token length (includes prompt).",
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Keep cache local-friendly when running on VMs
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    # Dayhoff (Jamba/Mamba) can require optional CUDA kernels. The model config supports
    # disabling them to fall back to the naive implementation.
    if hasattr(config, "use_mamba_kernels"):
        config.use_mamba_kernels = False

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()

    # Model card sample uses BOS only.
    inputs = tokenizer(
        tokenizer.bos_token,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    generations = []
    for _ in range(args.num_samples):
        output_ids = model.generate(
            **inputs,
            max_length=args.max_length,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        seq = tokenizer.batch_decode(output_ids.to("cpu"), skip_special_tokens=True)[0]
        generations.append(seq)

    for i, seq in enumerate(generations, start=1):
        print(f"=== SAMPLE {i} ===")
        print(seq)


if __name__ == "__main__":
    main()
