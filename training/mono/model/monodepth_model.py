import torch
import torch.nn as nn
from mono.utils.comm import get_func
from .__base_model__ import BaseDepthModel

# [新增] 引入 PEFT
try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    get_peft_model = None

class DepthModel(BaseDepthModel):
    def __init__(self, cfg, criterions, **kwards):
        super(DepthModel, self).__init__(cfg, criterions)   
        model_type = cfg.model.type
        self.training = True
        # print(self.__dict__.keys())
        # [新增] PEFT 注入逻辑
        # 我们通过检查 config 中是否有 'peft_config' 字段来决定是否启用
        if hasattr(cfg.model, 'peft_config') and cfg.model.peft_config is not None:
            if get_peft_model is None:
                raise ImportError("Please install peft: pip install peft")
            
            print(f"Injecting LoRA with config: {cfg.model.peft_config}")
            
            # 定义 LoRA 配置
            peft_config = LoraConfig(
                r=cfg.model.peft_config.get('r', 16),
                lora_alpha=cfg.model.peft_config.get('lora_alpha', 32),
                target_modules=cfg.model.peft_config.get('target_modules', ["qkv", "proj", "fc1", "fc2"]), # 针对 DINOv2 的关键层
                lora_dropout=cfg.model.peft_config.get('lora_dropout', 0.1),
                bias=cfg.model.peft_config.get('bias', "none"),
                # modules_to_save=[], # 如果需要解冻 Norm 层，可以在这里添加，或者后续手动解冻
            )
            
            # 使用 PEFT 包装 Backbone
            # 注意：BaseDepthModel 中通常将主干定义为 self.backbone
            self.depth_model = get_peft_model(self.depth_model, peft_config)
            
            # [重要] 打印可训练参数量，确认 LoRA 生效
            self.depth_model.print_trainable_parameters()
            
            # [针对 UAV 的特殊优化]
            # PEFT 默认会冻结所有非 LoRA 参数。
            # 对于无人机视角，建议手动解冻 LayerNorm 层以适应域偏移
            for name, param in self.depth_model.named_parameters():
                if "norm" in name or "ln" in name:
                    param.requires_grad = True
                    # print(f"Unfreezing {name}")
        
    # def inference(self, data):
    #     with torch.no_grad():
    #         pred_depth, _, confidence = self.inference(data)       
    #     return pred_depth, confidence

          
def get_monodepth_model(
    cfg : dict,
    criterions: dict,
    **kwargs
    ) -> nn.Module:
    # config depth  model
    model = DepthModel(cfg, criterions, **kwargs)
    #model.init_weights(load_imagenet_model, imagenet_ckpt_fpath)
    assert isinstance(model, nn.Module)
    return model


def get_configured_monodepth_model(
    cfg: dict,
    criterions: dict,
    ) -> nn.Module:
    """
        Args:
        @ configs: configures for the network.
        @ load_imagenet_model: whether to initialize from ImageNet-pretrained model.
        @ imagenet_ckpt_fpath: string representing path to file with weights to initialize model with.
        Returns:
        # model: depth model.
    """
    model = get_monodepth_model(cfg, criterions)
    return model


