import torch


def load_pretrained_weights(model, pretrained_path, module_name):
    """
    获取模型某层权重

    Args:
        model: 目标model
        pretrained_path (str): 预训练模型的路径
        module_name (str): 模块名字
    Returns:
        model: 更新后的模型
    """
    # Load the full pretrained state dict
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')

    if isinstance(pretrained_dict, dict) and 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']

    # 只保留需要的权重
    module_dict = {}
    for key, value in pretrained_dict.items():
        # print(f"pretrained_dict key: {key}")
        if "model" in key:
            for key, value in value.items():
                # print(f"model key: {key}")
                if module_name in key:
                    # Remove the module prefix if needed
                    # print("module_name:",module_name)
                    new_key = key.replace(f"{module_name}.", "")
                    module_dict[new_key] = value
    # 清理不需要的数据
    del pretrained_dict
    torch.cuda.empty_cache()

    # print(f"Found {module_dict} weights for module {module_name}")
    # 加载筛选后的权重
    target_module = getattr(model, module_name)
    # print(f"Loading pretrained weights for module {target_module}")
    # print(f"before target_module: {target_module.state_dict()}")
    target_module.load_state_dict(module_dict, strict=True)
    # print(f"after target_module: {target_module.state_dict()}")
    #  最后清理
    del module_dict
    torch.cuda.empty_cache()
    return model


def freeze_module(model, module_name):
    """
    冻结模型指定模块的参数

    Args:
        model: 目标模型
        module_name (str): 要冻结的模块名称
    Returns:
        model: 更新后的模型
    """
    target_module = getattr(model, module_name)
    # print(f"Freezing module {target_module}")
    # 冻结参数
    for param in target_module.parameters():
        param.requires_grad = False

    # 验证冻结状态
    frozen_params = sum(1 for p in target_module.parameters() if not p.requires_grad)
    total_params = sum(1 for _ in target_module.parameters())
    print(f"模块 {module_name} 已冻结 {frozen_params}/{total_params} 个参数")

    return model
