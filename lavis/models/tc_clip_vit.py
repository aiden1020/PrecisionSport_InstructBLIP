import torch
from pathlib import Path
from omegaconf import OmegaConf
from hydra import initialize, compose
from lavis.models.tc_clip_encoder.trainers.build_trainer import returnCLIP
from lavis.models.tc_clip_encoder.utils.logger import create_logger
from lavis.models.tc_clip_encoder.utils.tools import load_checkpoint


import torch
import torch.nn as nn


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

#         if isinstance(l, (nn.MultiheadAttention, Attention)):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()

    model.apply(_convert_weights_to_fp16)
# def load_tc_clip_model(config_path="tc_clip_encoder/configs", output="lavis/models/tc_clip_encoder/workspace/inference",
#                         tc_clip_model_path="lavis/models/tc_clip_encoder/weight/fully-supervised-name-class-22-86.pth", precision="fp16"):
def load_tc_clip_model(config_path="tc_clip_encoder/configs", output="lavis/models/tc_clip_encoder/workspace/inference",
                        tc_clip_model_path="lavis/models/tc_clip_encoder/weight/fully-supervised-LLM-class-22-85.pth", precision="fp16"):

    device = torch.device(f"cuda")
    
    overrides = [
        f"output={output}",
        "eval=test",
        "trainer=tc_clip",
        f"resume={tc_clip_model_path}"
    ]
    
    with initialize(version_base=None, config_path=config_path):
        config = compose(config_name="zero_shot.yaml", overrides=overrides)
    
    OmegaConf.set_struct(config, False)
    Path(config.output).mkdir(parents=True, exist_ok=True)
    
    config.trainer_name = "TCCLIP_encoder"
    logger = create_logger(output_dir=config.output, dist_rank=0, name=f"{config.trainer_name}")
    logger.disabled = True
    
    model = returnCLIP(config, logger).to(device)
    if config.resume:
        load_checkpoint(config, model, None, None, logger, model_only=True)
    if precision == "fp16":
#         model.to("cuda") 
        convert_weights_to_fp16(model)
    return model 
