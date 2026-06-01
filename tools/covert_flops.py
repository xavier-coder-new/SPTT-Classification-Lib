def convert_flops(value, target_unit=None):
    """
    将浮点运算次数转换为人类可读的单位 (GFLOPs, TFLOPs 等)
    
    :param value: 原始的 FLOPs 数值 (int 或 float)
    :param target_unit: 目标单位 (可选), 如 'G', 'T', 'P'。如果不填，会自动选择最合适的单位。
    :return: 格式化的字符串
    """
    if value == 0: return "0 FLOPs"

    # 定义单位阶梯 (以 1000 为进率，符合计算机性能度量标准)
    units = [
        (1e0, "FLOPs"),
        (1e3, "KFLOPs"),
        (1e6, "MFLOPs"),
        (1e9, "GFLOPs"),
        (1e12, "TFLOPs"),
        (1e15, "PFLOPs"),
        (1e18, "EFLOPs")
    ]

    # 1. 如果没有指定目标单位，自动寻找最合适的量级
    if target_unit is None:
        # 从大到小遍历，找到第一个小于该数值的单位
        for factor, name in reversed(units):
            if value >= factor:
                converted_value = value / factor
                return f"{converted_value:,.4f} {name}"
        return f"{value:,.0f} FLOPs"

    # 2. 如果指定了目标单位 (例如 'G' 或 'T')
    unit_map = {
        'F': 1e0, 'K': 1e3, 'M': 1e6, 
        'G': 1e9, 'T': 1e12, 'P': 1e15, 'E': 1e18
    }
    
    target_key = target_unit.upper()[0] # 提取首字母，兼容 'TFLOPS', 'T', 'tera' 等输入
    
    if target_key not in unit_map:
        return "错误：不支持的单位"

    factor = unit_map[target_key]
    converted_value = value / factor
    
    # 动态调整小数位数，避免显示过多的 0
    decimals = 6 if converted_value < 0.01 else 4
    return f"{converted_value:,.{decimals}f} {units[[k[0] for k in units].index(target_key)][1]}"

# --- 测试数据 ---
sptt_flops = 35888335872.0
bptt_flops = 35888335872.0

print(f"原始数值: {my_flops}")
print("-" * 30)

# 1. 自动模式 (推荐)
print(f"自动适配: {convert_flops(my_flops)}")

# 2. 强制指定单位
print(f"转为 GFLOPs: {convert_flops(my_flops, 'G')}")
print(f"转为 TFLOPs: {convert_flops(my_flops, 'T')}")
print(f"转为 MFLOPs: {convert_flops(my_flops, 'M')}")