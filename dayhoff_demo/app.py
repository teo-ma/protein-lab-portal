import argparse
import base64
import os
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio demo for Dayhoff-3b-GR-HM-c")
    parser.add_argument(
        "--model",
        default="microsoft/Dayhoff-3b-GR-HM-c",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Gradio server bind address (Container Apps: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "7860")),
        help="Gradio server port (Container Apps uses PORT env var)",
    )
    return parser.parse_args()


@lru_cache(maxsize=1)
def load_model_and_tokenizer(model_id: str):
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(config, "use_mamba_kernels"):
        config.use_mamba_kernels = False

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()

    return model, tokenizer


@lru_cache(maxsize=8)
def load_tokenizer_only(model_id: str):
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def get_bos_token_info(model_id: str) -> tuple[str, str]:
    tokenizer = load_tokenizer_only(model_id)
    bos = tokenizer.bos_token
    bos_id = tokenizer.bos_token_id
    return (repr(bos), str(bos_id))


def generate_sequence(
    prompt: str,
    seed: int,
    max_length: int,
    num_samples: int,
    temperature: float,
    top_p: float,
    model_id: str,
) -> str:
    set_seed(seed)

    model, tokenizer = load_model_and_tokenizer(model_id)
    device = next(model.parameters()).device

    prompt_text = prompt.strip()
    if not prompt_text:
        prompt_text = tokenizer.bos_token

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    results: list[str] = []
    for i in range(num_samples):
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        seq = tokenizer.batch_decode(output_ids.to("cpu"), skip_special_tokens=True)[0]
        results.append(f"=== SAMPLE {i + 1} ===\n{seq}")

    return "\n\n".join(results)


_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def extract_sequences_from_output(text: str) -> list[str]:
    sequences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("==="):
            continue
        if _AA_RE.match(line):
            sequences.append(line)
    return sequences


def bioemu_validate_sequences(
    generated_output: str,
    bioemu_num_samples: int,
    bioemu_model_name: str,
    filter_samples: bool,
    output_base_dir: str,
) -> tuple[str, str]:
    try:
        from bioemu.sample import main as bioemu_sample
        from bioemu.utils import count_samples_in_output_dir
    except Exception as exc:  # noqa: BLE001
        return (
            "BioEmu is not installed in this environment.\n"
            "Install on the VM: pip install bioemu\n\n"
            f"Import error: {exc!r}",
            "",
        )

    bioemu_model_name = (bioemu_model_name or "").strip()
    # The BioEmu python package expects a short model name like "bioemu-v1.1",
    # not the Hugging Face repo id.
    if not bioemu_model_name:
        bioemu_model_name = "bioemu-v1.1"
    if bioemu_model_name.lower() == "microsoft/bioemu":
        bioemu_model_name = "bioemu-v1.1"

    sequences = extract_sequences_from_output(generated_output)
    if not sequences:
        return ("No valid amino-acid sequences found to validate.", "")

    os.makedirs(output_base_dir, exist_ok=True)

    logs: list[str] = []
    logs.append(f"BioEmu validation for {len(sequences)} sequence(s)")
    logs.append(
        "Note: First-time BioEmu sampling may download and set up ColabFold; this can take a while."
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    last_out_dir = ""

    for idx, seq in enumerate(sequences, start=1):
        prefix = seq[:10]
        out_dir = os.path.join(output_base_dir, f"bioemu_{run_id}_s{idx}_{prefix}")
        os.makedirs(out_dir, exist_ok=True)
        last_out_dir = out_dir
        logs.append("")
        logs.append(f"=== BioEmu for SAMPLE {idx} (len={len(seq)}) ===")
        logs.append(f"output_dir: {out_dir}")

        before_count = None
        try:
            before_count = int(count_samples_in_output_dir(Path(out_dir)))
        except Exception:
            before_count = None

        t0 = time.time()
        try:
            # bioemu_sample supports both CLI and python API; we keep args minimal.
            bioemu_sample(
                sequence=seq,
                num_samples=int(bioemu_num_samples),
                output_dir=out_dir,
                model_name=bioemu_model_name,
                filter_samples=filter_samples,
            )
            dt = time.time() - t0

            after_count = None
            try:
                after_count = int(count_samples_in_output_dir(Path(out_dir)))
            except Exception:
                after_count = None

            logs.append(f"status: ok ({dt:.1f}s)")
            if after_count is not None:
                if before_count is None:
                    logs.append(f"samples_kept: {after_count} (requested: {int(bioemu_num_samples)})")
                else:
                    logs.append(
                        f"samples_kept: {after_count} (requested: {int(bioemu_num_samples)}, previous: {before_count}, new: {after_count - before_count})"
                    )
            else:
                logs.append(f"requested: {int(bioemu_num_samples)} (kept count unavailable)")

            topology_path = Path(out_dir) / "topology.pdb"
            xtc_path = Path(out_dir) / "samples.xtc"
            topology_ok = topology_path.is_file()
            xtc_ok = xtc_path.is_file()
            logs.append(
                f"files: topology.pdb={'ok' if topology_ok else 'missing'}, samples.xtc={'ok' if xtc_ok else 'missing'}"
            )
        except TypeError:
            # Backward/forward compatibility: if signature differs, retry with minimal args.
            t0 = time.time()
            bioemu_sample(sequence=seq, num_samples=int(bioemu_num_samples), output_dir=out_dir)
            dt = time.time() - t0
            after_count = None
            try:
                after_count = int(count_samples_in_output_dir(Path(out_dir)))
            except Exception:
                after_count = None

            logs.append(f"status: ok (fallback args, {dt:.1f}s)")
            if after_count is not None:
                logs.append(f"samples_kept: {after_count} (requested: {int(bioemu_num_samples)})")
            else:
                logs.append(f"requested: {int(bioemu_num_samples)} (kept count unavailable)")

            topology_path = Path(out_dir) / "topology.pdb"
            xtc_path = Path(out_dir) / "samples.xtc"
            topology_ok = topology_path.is_file()
            xtc_ok = xtc_path.is_file()
            logs.append(
                f"files: topology.pdb={'ok' if topology_ok else 'missing'}, samples.xtc={'ok' if xtc_ok else 'missing'}"
            )
        except Exception as exc:  # noqa: BLE001
            dt = time.time() - t0
            logs.append(f"status: error ({dt:.1f}s): {exc!r}")

    return ("\n".join(logs), last_out_dir)


def _read_text_limited(path: Path, max_bytes: int = 2_000_000) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


_BIOEMU_HEAD = r"""
<script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
<script>
(() => {
    const READY_ATTR = "data-bioemu-initialized";

    const countFramesFromPdbText = (text) => {
        const endmdl = (text.match(/\nENDMDL\b/g) || []).length;
        if (endmdl > 0) return endmdl;
        const model = (text.match(/\nMODEL\s+\d+/g) || []).length;
        if (model > 0) return model;
        return 1;
    };

    const waitFor3Dmol = (retries = 200) => new Promise((resolve, reject) => {
        const step = () => {
            if (window.$3Dmol && window.$3Dmol.createViewer) return resolve();
            if (retries-- <= 0) return reject(new Error("3Dmol.js failed to load"));
            setTimeout(step, 50);
        };
        step();
    });

    const initOne = async (root) => {
        if (!root || root.getAttribute(READY_ATTR) === "1") return;
        root.setAttribute(READY_ATTR, "1");

        const frameEl = root.querySelector(".bioemu_frame");
        const labelEl = root.querySelector(".bioemu_frame_label");
        const viewerEl = root.querySelector(".bioemu_viewer");
        const b64El = root.querySelector(".bioemu_pdb_b64");
        if (!frameEl || !labelEl || !viewerEl || !b64El) return;

        const pdbB64 = (b64El.value || "").trim();
        const pdb = pdbB64 ? atob(pdbB64) : "";
        let frameCount = Math.max(1, parseInt(root.getAttribute("data-frame-count") || "1", 10) || 1);
        frameCount = Math.max(frameCount, countFramesFromPdbText(pdb));

        const refreshUi = () => {
            const max = Math.max(0, frameCount - 1);
            frameEl.max = String(max);
            let v = parseInt(frameEl.value, 10);
            if (!Number.isFinite(v)) v = 0;
            if (v < 0) v = 0;
            if (v > max) v = max;
            frameEl.value = String(v);
            labelEl.textContent = `${v}/${max}`;
        };

        refreshUi();

        let viewer = null;
        let timer = null;
        const stop = () => {
            if (timer !== null) {
                clearInterval(timer);
                timer = null;
            }
            if (viewer && viewer.stopAnimate) viewer.stopAnimate();
        };

        const setFrame = (f) => {
            if (!viewer || !viewer.setFrame) return;
            viewer.setFrame(f);
            viewer.render();
        };

        const tick = () => {
            if (!viewer || !viewer.setFrame) return;
            if (frameCount <= 1) return;
            const cur = parseInt(frameEl.value, 10) || 0;
            const next = (cur + 1) % frameCount;
            frameEl.value = String(next);
            setFrame(next);
            refreshUi();
        };

        frameEl.addEventListener("input", () => {
            const f = parseInt(frameEl.value, 10) || 0;
            setFrame(f);
            refreshUi();
        });

        const playBtn = root.querySelector(".bioemu_play");
        const stopBtn = root.querySelector(".bioemu_stop");
        if (playBtn) {
            playBtn.addEventListener("click", () => {
                stop();
                if (!viewer || !viewer.setFrame || frameCount <= 1) return;
                timer = setInterval(tick, 250);
            });
        }
        if (stopBtn) stopBtn.addEventListener("click", () => stop());

        try {
            await waitFor3Dmol();
            viewer = window.$3Dmol.createViewer(viewerEl, { backgroundColor: "white" });
            if (viewer.addModelsAsFrames) {
                viewer.addModelsAsFrames(pdb, "pdb");
            } else if (viewer.addModel) {
                viewer.addModel(pdb, "pdb");
            }
            viewer.setStyle({}, { cartoon: {} });
            viewer.zoomTo();
            viewer.render();

            setTimeout(() => {
                try {
                    const n = (viewer.getNumFrames && viewer.getNumFrames()) || (viewer.getNumModels && viewer.getNumModels()) || 0;
                    if (n && n > frameCount) frameCount = n;
                } catch (e) {}
                refreshUi();
            }, 0);
            setTimeout(() => {
                try {
                    const n = (viewer.getNumFrames && viewer.getNumFrames()) || (viewer.getNumModels && viewer.getNumModels()) || 0;
                    if (n && n > frameCount) frameCount = n;
                } catch (e) {}
                refreshUi();
            }, 200);

            refreshUi();
        } catch (e) {
            console.error(e);
            labelEl.textContent = "viewer init failed";
        }
    };

    const initAll = () => {
        document.querySelectorAll(".bioemu_root").forEach((root) => {
            if (root.getAttribute(READY_ATTR) !== "1") initOne(root);
        });
    };

    initAll();
    const obs = new MutationObserver(() => initAll());
    obs.observe(document.documentElement, { childList: true, subtree: true });
})();
</script>
"""


def _build_3dmol_html(pdb_b64: str, frame_count: int) -> str:
    # NOTE: gr.HTML inserts content via innerHTML; embedded <script> will not reliably execute.
    # We return pure DOM and rely on the global JS injected via demo.launch(head=...).
    import secrets

    key = secrets.token_hex(8)
    safe_frames = int(max(1, frame_count))
    return (
        f"<div class=\"bioemu_root\" data-bioemu-key=\"{key}\" data-frame-count=\"{safe_frames}\">"
        "<div style=\"display:flex; gap:12px; align-items:center; margin: 8px 0;\">"
        "<button class=\"bioemu_play\">Play</button>"
        "<button class=\"bioemu_stop\">Stop</button>"
        "<label style=\"margin-left:8px;\">Frame</label>"
        "<input class=\"bioemu_frame\" type=\"range\" min=\"0\" max=\"0\" value=\"0\" step=\"1\" style=\"width: 320px;\" />"
        "<span class=\"bioemu_frame_label\">0/0</span>"
        "</div>"
        "<div class=\"bioemu_viewer\" style=\"width: 100%; height: 520px; position: relative; border: 1px solid #ddd;\"></div>"
        f"<textarea class=\"bioemu_pdb_b64\" style=\"display:none;\">{pdb_b64}</textarea>"
        "</div>"
    )


def _count_models_in_pdb_text(text: str) -> int:
    endmdl = text.count("\nENDMDL")
    if endmdl > 0:
        return endmdl
    model = len(re.findall(r"\nMODEL\s+\d+", text))
    if model > 0:
        return model
    return 1


def bioemu_visualize_output(
    output_dir: str,
    max_frames: int,
    stride: int,
) -> tuple[str, str, list[str]]:
    output_dir = (output_dir or "").strip()
    if not output_dir:
        return ("Please provide a BioEmu output_dir.", "", [])

    try:
        import MDAnalysis as mda
    except Exception as exc:  # noqa: BLE001
        return (
            "MDAnalysis is not installed in this environment.\n"
            "This feature is available on Linux with the BioEmu dependencies installed.\n\n"
            f"Import error: {exc!r}",
            "",
            [],
        )

    out_dir = Path(output_dir).expanduser()
    if not out_dir.exists():
        return (f"output_dir does not exist: {out_dir}", "", [])

    topology_path = out_dir / "topology.pdb"
    xtc_path = out_dir / "samples.xtc"
    if not topology_path.is_file() or not xtc_path.is_file():
        return (
            "Missing required files in output_dir.\n"
            f"- topology.pdb: {'ok' if topology_path.is_file() else 'missing'}\n"
            f"- samples.xtc: {'ok' if xtc_path.is_file() else 'missing'}\n",
            "",
            [],
        )

    stride = max(1, int(stride))
    max_frames = max(1, int(max_frames))

    logs: list[str] = []
    logs.append(f"output_dir: {out_dir}")
    logs.append(f"topology: {topology_path.name}")
    logs.append(f"trajectory: {xtc_path.name}")

    try:
        u = mda.Universe(str(topology_path), str(xtc_path))
    except Exception as exc:  # noqa: BLE001
        return (f"Failed to load trajectory: {exc!r}", "", [str(topology_path), str(xtc_path)])

    try:
        traj_frames = int(len(u.trajectory))
    except Exception:
        traj_frames = -1

    logs.append(f"trajectory_frames: {traj_frames}")

    preview_path = out_dir / f"trajectory_preview_max{max_frames}_stride{stride}.pdb"

    written = 0
    preview_text = ""
    preview_models = 1
    preview_bytes = 0
    truncated = False

    target_max_frames = max_frames
    for _attempt in range(4):
        written = 0
        writer = mda.Writer(str(preview_path), multiframe=True)
        try:
            for _ts in u.trajectory[::stride]:
                writer.write(u.atoms)
                written += 1
                if written >= target_max_frames:
                    break
        finally:
            writer.close()

        preview_bytes = preview_path.stat().st_size if preview_path.exists() else 0
        preview_text = _read_text_limited(preview_path)
        truncated = preview_bytes > 2_000_000
        preview_models = _count_models_in_pdb_text(preview_text)

        if truncated and written > 1 and preview_models <= 1 and target_max_frames > 1:
            target_max_frames = max(1, target_max_frames // 2)
            continue
        break

    logs.append(f"preview_pdb: {preview_path.name}")
    logs.append(f"preview_frames: {written} (stride={stride})")
    logs.append(f"preview_bytes: {preview_bytes}")
    logs.append(f"embed_truncated: {truncated}")
    logs.append(f"embed_models: {preview_models}")

    pdb_b64 = base64.b64encode(preview_text.encode("utf-8")).decode("ascii")
    html = _build_3dmol_html(pdb_b64, frame_count=preview_models)
    files = [str(topology_path), str(xtc_path), str(preview_path)]
    return ("\n".join(logs), html, files)


def build_ui(default_model: str) -> gr.Blocks:
    here_dir = Path(__file__).resolve().parent
    repo_root = here_dir.parent
    local_images_dir = here_dir / "images"
    root_images_dir = repo_root / "images"
    banner_candidates = [root_images_dir / "banner.png", local_images_dir / "banner.png"]

    def _png_data_uri(path: Path) -> str:
        try:
            data_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        except Exception:
            return ""
        return f"data:image/png;base64,{data_b64}"

    def _first_existing_path(paths: list[Path]) -> Path | None:
        for path in paths:
            if path.is_file():
                return path
        return None

    with gr.Blocks(title="Dayhoff + BioEmu蛋白质链实验室") as demo:
        banner_path = _first_existing_path(banner_candidates)
        if banner_path is not None:
            banner_uri = _png_data_uri(banner_path)
            if banner_uri:
                gr.Markdown(
                    f'<img src="{banner_uri}" alt="banner" style="width:100%; height:auto;" />'
                )

        gr.Markdown(
            "<h1 style=\"text-align:center; margin: 0.2rem 0 0.8rem 0;\">Dayhoff + BioEmu蛋白质链实验室</h1>\n"
            "Dayhoff + BioEmu 组合用于构建闭环的蛋白质设计与验证工作流："
            "**生成序列 → 结构采样/验证 → 可视化预览**。\n\n"
            "- Dayhoff：快速探索序列空间，生成候选蛋白序列\n"
            "- BioEmu：对候选序列进行结构集合（ensemble）采样与筛选（可选）"
        )

        model_id = gr.Textbox(
            label="模型（Hugging Face ID 或本地路径）",
            value=default_model,
        )
        prompt = gr.Textbox(
            label="提示词（可选）",
            placeholder="留空则使用 BOS token",
            lines=2,
        )

        gr.Examples(
            label="示例提示词",
            examples=[
                [""],
                ["M"],
                ["MA"],
                ["MKTIIALSYIFCLVFA"],
                ["ACDEFGHIKLMNPQRSTVWY"],
            ],
            inputs=[prompt],
        )

        with gr.Row():
            seed = gr.Number(label="随机种子（Seed）", value=0, precision=0)
            max_length = gr.Slider(label="最大长度（Max length）", minimum=16, maximum=512, value=50, step=1)
            num_samples = gr.Slider(label="生成条数（Samples）", minimum=1, maximum=8, value=1, step=1)

        with gr.Row():
            temperature = gr.Slider(label="温度（Temperature）", minimum=0.1, maximum=2.0, value=1.0, step=0.05)
            top_p = gr.Slider(label="Top-p", minimum=0.1, maximum=1.0, value=1.0, step=0.05)

        with gr.Row():
            bos_btn = gr.Button("查看 BOS token")
            bos_token = gr.Textbox(label="BOS token（repr）", interactive=False)
            bos_token_id = gr.Textbox(label="BOS token id", interactive=False)

        run_btn = gr.Button("生成蛋白序列")
        output = gr.Textbox(label="生成结果", lines=10)

        gr.Markdown(
            "## BioEmu 验证（可选）\n"
            "对生成序列进行结构集合（ensemble）采样，并输出可用于可视化/下载的结果文件。"
        )

        with gr.Row():
            bioemu_num_samples = gr.Slider(label="BioEmu 采样数（num_samples）", minimum=1, maximum=50, value=5, step=1)
            bioemu_model_name = gr.Textbox(label="BioEmu 模型名（model_name）", value="bioemu-v1.1")
            filter_samples = gr.Checkbox(label="过滤非物理样本（Filter unphysical samples）", value=True)

        output_base_dir = gr.Textbox(
            label="BioEmu 输出根目录（output base dir）",
            value=os.environ.get("BIOEMU_OUTPUT_BASE_DIR", "/tmp/bioemu_runs"),
        )
        bioemu_btn = gr.Button("用 BioEmu 验证")
        bioemu_output = gr.Textbox(label="BioEmu 验证日志", lines=12)

        gr.Markdown(
            "## BioEmu 可视化（Web）\n"
            "将验证日志中的 `output_dir` 粘贴到下方输入框，即可在浏览器内预览结构集合并播放。"
        )
        viz_output_dir = gr.Textbox(label="BioEmu 输出目录（output_dir）", placeholder="/mnt/data/bioemu_runs/bioemu_...", lines=1)
        with gr.Row():
            viz_max_frames = gr.Slider(label="预览最大帧数（max frames）", minimum=1, maximum=200, value=50, step=1)
            viz_stride = gr.Slider(label="帧步长（stride）", minimum=1, maximum=50, value=1, step=1)
        viz_btn = gr.Button("渲染 Web 预览")
        viz_log = gr.Textbox(label="可视化日志", lines=6)
        viz_view = gr.HTML(label="3D 预览")
        viz_files = gr.Files(label="下载文件")

        bos_btn.click(
            fn=get_bos_token_info,
            inputs=[model_id],
            outputs=[bos_token, bos_token_id],
        )

        run_btn.click(
            fn=generate_sequence,
            inputs=[prompt, seed, max_length, num_samples, temperature, top_p, model_id],
            outputs=[output],
        )

        bioemu_btn.click(
            fn=bioemu_validate_sequences,
            inputs=[output, bioemu_num_samples, bioemu_model_name, filter_samples, output_base_dir],
            outputs=[bioemu_output, viz_output_dir],
        )

        viz_btn.click(
            fn=bioemu_visualize_output,
            inputs=[viz_output_dir, viz_max_frames, viz_stride],
            outputs=[viz_log, viz_view, viz_files],
        )

    return demo


def main() -> None:
    args = parse_args()
    demo = build_ui(args.model)
    demo.queue(status_update_rate=1, default_concurrency_limit=1)

    # Allow downloading files from the BioEmu output directory.
    output_base_dir = os.environ.get("BIOEMU_OUTPUT_BASE_DIR", "/tmp/bioemu_runs")
    images_dir = str(Path(__file__).resolve().parent / "images")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        head=_BIOEMU_HEAD,
        allowed_paths=[output_base_dir, "/tmp", images_dir],
    )


if __name__ == "__main__":
    main()
