"""
pixeliar — ModelLoader (Module 6)
Load all ML models to GPU VRAM, log sizes, warm-up pass.
Handles missing weights gracefully (log + skip).
"""

import os
import sys
import torch

# ── Compatibility patches for Python 3.12+ and newer torchvision ──

# Python 3.12 removed the `imp` module — some ML repos still use it
if "imp" not in sys.modules:
    import importlib
    import types
    _imp = types.ModuleType("imp")
    _imp.find_module = lambda *a, **k: None
    _imp.load_module = lambda *a, **k: None
    sys.modules["imp"] = _imp

# Retinexformer/basicsr imports torchvision.transforms.functional_tensor which was removed
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as _F
    sys.modules["torchvision.transforms.functional_tensor"] = _F


def _count_params(model):
    return sum(p.numel() for p in model.parameters())


def _vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def _warmup(model, name, logger):
    """Dummy forward pass to ensure GPU is ready."""
    try:
        dummy = torch.zeros(1, 3, 64, 64).cuda()
        with torch.no_grad():
            if name == "deepwb":
                model(dummy)
            elif name in ("iat", "retinex"):
                model(dummy)
            elif name == "csrnet":
                model(dummy)
            elif name == "restormer":
                model(dummy)
        logger.p("STEP", f"  warm-up {name} OK", indent=1)
    except Exception as e:
        logger.p("WARN", f"  warm-up {name} failed: {e}", indent=1)


def _load_iat(config, logger):
    """IAT — Illumination Adaptive Transformer."""
    repo_dir = os.path.join(config["weights_dir"], "IAT")
    if not os.path.exists(repo_dir):
        import subprocess
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/cuiziteng/Illumination-Adaptive-Transformer",
            repo_dir,
        ], check=True)

    # IAT code lives inside IAT_enhance/ subdir
    iat_dir = os.path.join(repo_dir, "IAT_enhance")
    sys.path.insert(0, iat_dir)
    from model.IAT_main import IAT

    model = IAT().cuda().eval()
    weight_path = os.path.join(iat_dir, "best_Epoch_lol_v1.pth")
    model.load_state_dict(torch.load(weight_path, map_location="cuda"))
    return model


def _load_retinex(config, logger):
    """Retinexformer — severe low-light."""
    repo_dir = os.path.join(config["weights_dir"], "Retinexformer")
    if not os.path.exists(repo_dir):
        import subprocess
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/caiyuanhao1998/Retinexformer",
            repo_dir,
        ], check=True)

    # Import arch directly by file path (bypasses broken basicsr package registration)
    import importlib.util
    arch_path = os.path.join(repo_dir, "basicsr", "models", "archs", "RetinexFormer_arch.py")
    spec = importlib.util.spec_from_file_location("RetinexFormer_arch", arch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    RetinexFormer = mod.RetinexFormer

    model = RetinexFormer(
        in_channels=3, out_channels=3, n_feat=40, stage=1,
        num_blocks=[1, 2, 2],
    ).cuda().eval()
    weight_path = os.path.join(repo_dir, "LOL_v1.pth")
    if not os.path.exists(weight_path):
        # Use Drive API to find and download from shared folder
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        from google.auth import default
        creds, _ = default()
        svc = build("drive", "v3", credentials=creds)

        # Search in the Retinexformer shared weights folder
        folder_id = "1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV"
        results = svc.files().list(
            q=f"'{folder_id}' in parents and name contains 'LOL_v1' and trashed=false",
            fields="files(id, name)",
        ).execute()
        files = results.get("files", [])
        if not files:
            raise FileNotFoundError("LOL_v1.pth not found in Retinexformer shared folder")

        request = svc.files().get_media(fileId=files[0]["id"])
        with open(weight_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
    model.load_state_dict(torch.load(weight_path, map_location="cuda")["params"])
    return model


def _load_deepwb(config, logger):
    """Deep White Balance."""
    repo_dir = os.path.join(config["weights_dir"], "DeepWB")
    if not os.path.exists(repo_dir):
        import subprocess
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/mahmoudnafifi/Deep_White_Balance",
            repo_dir,
        ], check=True)

    # DeepWB code lives inside PyTorch/ subdir
    pytorch_dir = os.path.join(repo_dir, "PyTorch")
    sys.path.insert(0, pytorch_dir)
    from arch.deep_wb_model import deepWBNet

    model = deepWBNet().cuda().eval()
    weight_path = os.path.join(pytorch_dir, "models", "net.pth")
    model.load_state_dict(torch.load(weight_path, map_location="cuda"))
    return model


def _load_csrnet(config, logger):
    """CSRNet — professional retouching."""
    repo_dir = os.path.join(config["weights_dir"], "CSRNet")
    if not os.path.exists(repo_dir):
        import subprocess
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/hejingwenhejingwen/CSRNet",
            repo_dir,
        ], check=True)

    sys.path.insert(0, repo_dir)
    from codes.models.archs.CSRNet_arch import CSRNet as CSRNetArch

    model = CSRNetArch().cuda().eval()
    weight_path = os.path.join(repo_dir, "experiments", "pretrain_models", "csrnet.pth")
    model.load_state_dict(torch.load(weight_path, map_location="cuda"))
    return model


def _load_nafnet(config, logger):
    """NAFNet — denoise + deblur via nafnetlib."""
    import subprocess
    subprocess.run(["pip", "install", "-q", "nafnetlib"], check=True)
    from nafnetlib import DenoiseProcessor, DeblurProcessor

    model_dir = config.get("nafnet_model_dir", "/content/pixeliar_nafnet_weights")
    os.makedirs(model_dir, exist_ok=True)

    denoise = DenoiseProcessor(model_id="sidd_width64", model_dir=model_dir, device="cuda")
    deblur = DeblurProcessor(model_id="reds_width64", model_dir=model_dir, device="cuda")
    return denoise, deblur


def _load_restormer(config, logger):
    """Restormer — heavy combined degradation (optional)."""
    if not config.get("load_restormer", False):
        return None

    repo_dir = os.path.join(config["weights_dir"], "Restormer")
    if not os.path.exists(repo_dir):
        import subprocess
        subprocess.run(["pip", "install", "-q", "basicsr"], check=True)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/swz30/Restormer",
            repo_dir,
        ], check=True)

    sys.path.insert(0, repo_dir)
    from basicsr.models.archs.restormer_arch import Restormer

    model = Restormer(
        inp_channels=3, out_channels=3,
        dim=48, num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66, bias=False,
        LayerNorm_type="BiasFree",
        dual_pixel_task=False,
    ).cuda().eval()

    weight_path = os.path.join(repo_dir, "pretrained_models", "real_denoising.pth")
    if not os.path.exists(weight_path):
        import gdown
        gdown.download(id="1SHMBoy5YMSck1G1i7kCNYaWu0vNR3Sws", output=weight_path, quiet=True)
    ckpt = torch.load(weight_path, map_location="cuda")
    model.load_state_dict(ckpt["params"])
    return model


def load_all_models(config, logger):
    """Load all models to VRAM. Returns dict of loaded models."""
    models = {}
    vram_start = _vram_mb()

    logger.p("MODEL", "Loading models to VRAM...", indent=1)

    # IAT
    try:
        m = _load_iat(config, logger)
        models["iat"] = m
        logger.model_loaded("IAT", _count_params(m), _vram_mb() - vram_start)
    except Exception as e:
        logger.p("FAIL", f"IAT load failed: {e}", indent=1)

    # Retinexformer
    try:
        m = _load_retinex(config, logger)
        models["retinex"] = m
        logger.model_loaded("Retinexformer", _count_params(m), _vram_mb() - vram_start)
    except Exception as e:
        logger.p("FAIL", f"Retinexformer load failed: {e}", indent=1)

    # DeepWB
    try:
        m = _load_deepwb(config, logger)
        models["deepwb"] = m
        logger.model_loaded("DeepWB", _count_params(m), _vram_mb() - vram_start)
    except Exception as e:
        logger.p("FAIL", f"DeepWB load failed: {e}", indent=1)

    # CSRNet
    try:
        m = _load_csrnet(config, logger)
        models["csrnet"] = m
        logger.model_loaded("CSRNet", _count_params(m), _vram_mb() - vram_start)
    except Exception as e:
        logger.p("FAIL", f"CSRNet load failed: {e}", indent=1)

    # NAFNet
    try:
        denoise, deblur = _load_nafnet(config, logger)
        models["naf_denoise"] = denoise
        models["naf_deblur"] = deblur
        logger.model_loaded("NAFNet-SIDD", 67_000_000, _vram_mb() - vram_start)
        logger.model_loaded("NAFNet-REDS", 67_000_000, _vram_mb() - vram_start)
    except Exception as e:
        logger.p("FAIL", f"NAFNet load failed: {e}", indent=1)

    # Restormer (optional)
    try:
        m = _load_restormer(config, logger)
        if m is not None:
            models["restormer"] = m
            logger.model_loaded("Restormer", _count_params(m), _vram_mb() - vram_start)
    except Exception as e:
        logger.p("FAIL", f"Restormer load failed: {e}", indent=1)

    # Summary
    total_vram = _vram_mb()
    if torch.cuda.is_available():
        total_avail = torch.cuda.get_device_properties(0).total_memory / 1024**2
        logger.p("VRAM",
                  f"Total VRAM used: {total_vram:.0f}MB / {total_avail:.0f}MB "
                  f"({total_vram / total_avail * 100:.1f}%)", indent=1)
    logger.p("MODEL", f"Loaded {len(models)} models", indent=1)

    return models
