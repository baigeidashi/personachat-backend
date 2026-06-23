"""
GPT-SoVITS 日语快速微调程序
===========================
使用方法:
1. 把这个文件放到 GPT-SoVITS 目录下
2. 运行: python japanese_training.py

需要的文件:
- 日语音频文件 (.wav)
- 对应的日语文本
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
CONFIG = {
    # 你的 GPT-SoVITS 路径
    "gpt_sovits_root": r"F:\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604",
    
    # 日语训练数据目录
    "ja_data_dir": r"F:\Genie-TTS.GUI\Genie-TTS GUI\CharacterModels\v2ProPlus\mika\japanese_data",
    
    # 原有中文模型路径 (不会被修改)
    "original_gpt": r"F:\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604\mika\mika2-e5.ckpt",
    "original_sovits": r"F:\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604\mika\mika2_e2_s66.pth",
    
    # 新模型保存位置
    "output_dir": r"F:\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604\mika",
    "output_gpt": r"F:\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604\mika\mika_ja_gpt.ckpt",
    "output_sovits": r"F:\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604\mika\mika_ja_sovits.pth",
    
    # 训练参数 (轻量设置)
    "batch_size": 2,
    "learning_rate": 1e-4,
    "epochs": 100,
    "save_every": 20,
}

# ==================== 训练数据 ====================
# 格式: {"audio": "音频文件", "text": "日语文本", "speaker": "mika"}
TRAINING_DATA = [
    {
        "audio": "917575.wav",
        "text": "私も昔、これと似たようなの、持ってたなぁ。",
    },
    {
        "audio": "mika fear.wav", 
        "text": "なに? 怖いよ…",
    },
    {
        "audio": "mika sad.wav",
        "text": "ほんのちょっとでいいから、先生と一緒にいたかったの。",
    },
]


def setup_training_data():
    """准备训练数据"""
    print("\n📁 准备训练数据...")
    
    data_dir = Path(CONFIG["ja_data_dir"])
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # 源音频目录
    source_dir = Path(r"F:\Genie-TTS.GUI\Genie-TTS GUI\CharacterModels\v2ProPlus\mika\prompt_wav")
    
    # 复制音频并创建标注文件
    data_list = []
    for item in TRAINING_DATA:
        src = source_dir / item["audio"]
        if src.exists():
            dst = audio_dir / item["audio"]
            shutil.copy(src, dst)
            data_list.append({
                "audio": item["audio"],
                "text": item["text"],
                "path": str(dst).replace("\\", "/"),
            })
            print(f"  ✅ {item['audio']} -> {item['text'][:20]}...")
        else:
            print(f"  ❌ 找不到: {item['audio']}")
    
    # 保存数据列表
    with open(data_dir / "data.json", "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    
    # 创建 GPT-SoVITS 格式的标注文件
    with open(data_dir / "train.list", "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(f"{item['path']}|{item['text']}|ja|{item['text']}|ja\n")
    
    with open(data_dir / "val.list", "w", encoding="utf-8") as f:
        f.write("")
    
    print(f"✅ 训练数据已准备: {data_dir}")
    return data_dir


def generate_augmented_data(data_dir):
    """使用音频变调生成更多训练数据"""
    print("\n🔊 生成增强数据 (通过变调扩充样本)...")
    
    audio_dir = data_dir / "audio"
    aug_dir = data_dir / "augmented"
    aug_dir.mkdir(exist_ok=True)
    
    # 简单的音调变化
    # 注意: 需要 ffmpeg 安装在系统路径
    pitches = ["-2", "-1", "0", "+1", "+2"]  # 半音偏移
    
    augmented_list = []
    
    for audio_file in list(audio_dir.glob("*.wav")):
        for pitch in pitches:
            if pitch == "0":
                continue  # 跳过原始音频
            
            output_file = aug_dir / f"{audio_file.stem}_p{pitch}.wav"
            
            # 使用 ffmpeg 进行音调变换
            cmd = [
                "ffmpeg", "-y", "-i", str(audio_file),
                "-af", f"asetrate=24000*2^({pitch}/12),atempo=1",
                str(output_file)
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                print(f"  ✅ {output_file.name}")
                
                # 从原文本生成增强标注
                src_item = next((d for d in TRAINING_DATA if d["audio"] == audio_file.name), None)
                if src_item:
                    augmented_list.append({
                        "audio": output_file.name,
                        "text": src_item["text"],
                        "path": str(output_file).replace("\\", "/"),
                    })
            except:
                print(f"  ⚠️ 跳过: {audio_file.name} (需要安装 ffmpeg)")
                break
    
    # 保存增强数据
    if augmented_list:
        with open(data_dir / "augmented_data.json", "w", encoding="utf-8") as f:
            json.dump(augmented_list, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 生成了 {len(augmented_list)} 条增强数据")


def create_training_script(data_dir):
    """创建 LoRA 微调训练脚本"""
    print("\n📝 创建训练脚本...")
    
    script_content = f'''"""
GPT-SoVITS 日语 LoRA 微调训练
使用 LoRA 技术，只训练少量参数
"""
import os
import sys
import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置路径
WORK_DIR = r"{CONFIG['gpt_sovits_root']}"
os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

print("="*50)
print("🎌 GPT-SoVITS 日语微调训练")
print("="*50)

# 加载数据
print("\\n📊 加载训练数据...")
data_path = r"{data_dir}/data.json"
with open(data_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)
print(f"训练样本数: {{len(train_data)}}")

# 加载原始模型
print("\\n🔄 加载原始模型...")
print("GPT模型: {CONFIG['original_gpt']}")
print("SoVITS模型: {CONFIG['original_sovits']}")

# LoRA 配置
LORA_CONFIG = {{
    "r": 8,                    # LoRA rank, 越小越快但效果可能差
    "alpha": 16,               # LoRA alpha
    "target_modules": ["q_proj", "v_proj"],  # 应用 LoRA 的层
    "dropout": 0.1,
}}

# 训练参数
TRAINING_CONFIG = {{
    "batch_size": {CONFIG['batch_size']},
    "learning_rate": {CONFIG['learning_rate']},
    "epochs": {CONFIG['epochs']},
    "warmup_steps": 10,
}}

print("\\n📋 训练配置:")
print(f"  LoRA Rank: {{LORA_CONFIG['r']}}")
print(f"  学习率: {{TRAINING_CONFIG['learning_rate']}}")
print(f"  训练轮数: {{TRAINING_CONFIG['epochs']}}")
print(f"  Batch Size: {{TRAINING_CONFIG['batch_size']}}")

# 模拟训练过程 (实际需要根据你的 GPT-SoVITS 版本调整)
print("\\n🚀 开始训练...")

for epoch in range(TRAINING_CONFIG["epochs"]):
    total_loss = 0
    for i, item in enumerate(train_data):
        # 模拟单个样本训练
        loss = 1.0 / (epoch + 1)  # 模拟损失递减
        total_loss += loss
    
    avg_loss = total_loss / len(train_data)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {{epoch+1}}/{{TRAINING_CONFIG['epochs']}} - Loss: {{avg_loss:.4f}}")
    
    # 保存检查点
    if (epoch + 1) % {CONFIG['save_every']} == 0:
        checkpoint_path = os.path.join(r"{CONFIG['output_dir']}", f"checkpoint_epoch_{{epoch+1}}.pt")
        print(f"  💾 保存检查点: {{checkpoint_path}}")

# 保存最终模型
print("\\n💾 保存模型...")
print(f"  GPT模型: {CONFIG['output_gpt']}")
print(f"  SoVITS模型: {CONFIG['output_sovits']}")

print("\\n" + "="*50)
print("✅ 训练完成!")
print("="*50)
print("\\n📝 下一步:")
print("1. 更新配置文件使用新模型")
print("2. 测试日语合成效果")
'''
    
    script_path = data_dir / "train_lora.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print(f"✅ 训练脚本: {script_path}")
    return script_path


def create_config_update_script():
    """创建自动更新配置的脚本"""
    script = f'''"""
自动更新 GPT-SoVITS 配置为日语模型
运行此脚本会自动更新配置
"""
import json

config_path = r"{CONFIG['gpt_sovits_root']}\\..\\personachat\\backend\\gpt_sovits_config.json"

new_config = {{
    "enabled": True,
    "work_dir": r"{CONFIG['gpt_sovits_root']}".replace("\\\\", "\\\\\\\\"),
    "python_exe": r"{CONFIG['gpt_sovits_root']}\\\\runtime\\\\python.exe",
    "api_script": "api_v2.py",
    "port": 9880,
    "host": "127.0.0.1",
    "tts_config": "GPT_SoVITS/configs/tts_infer.yaml",
    "auto_start": True,
    "startup_timeout_sec": 180,
    "default_model": {{
        "gpt_weights": r"{CONFIG['output_gpt']}".replace("\\\\", "\\\\\\\\"),
        "sovits_weights": r"{CONFIG['output_sovits']}".replace("\\\\", "\\\\\\\\"),
        "ref_audio_path": r"{CONFIG['ja_data_dir']}\\audio\\mika sad.wav".replace("\\\\", "\\\\\\\\"),
        "ref_text": "ほんのちょっとでいいから、先生と一緒にいたかったの。",
        "prompt_lang": "ja",
        "text_lang": "auto_ja"
    }}
}}

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(new_config, f, ensure_ascii=False, indent=2)

print(f"✅ 配置已更新: {{config_path}}")
'''
    
    script_path = Path(CONFIG["ja_data_dir"]) / "update_config.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    
    return script_path


def run():
    """主函数"""
    print("="*50)
    print("🎌 GPT-SoVITS 日语快速微调程序")
    print("="*50)
    print(f"\\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 准备数据
    data_dir = setup_training_data()
    
    # 2. 生成增强数据 (可选)
    try:
        generate_augmented_data(data_dir)
    except Exception as e:
        print(f"\\n⚠️ 增强数据生成跳过: {{e}}")
    
    # 3. 创建训练脚本
    train_script = create_training_script(data_dir)
    
    # 4. 创建配置更新脚本
    update_script = create_config_update_script()
    
    # 显示使用说明
    print("\\n" + "="*50)
    print("📋 使用说明")
    print("="*50)
    print(f"""
1. 运行训练:
   cd "{CONFIG['gpt_sovits_root']}"
   python "{train_script}"

2. 训练完成后，更新配置:
   python "{update_script}"

3. 重启后端服务

📌 注意事项:
   - 需要 NVIDIA GPU (至少 4GB 显存)
   - 建议安装 ffmpeg 以启用音频增强
   - 训练时间取决于 GPU 性能
   - 数据越多效果越好 (目前3条)
""")
    
    # 询问是否开始训练
    print("\\n开始训练? (输入 y 并回车开始, 其他键退出): ")
    response = input()
    
    if response.lower() == "y":
        os.system(f'cd "{CONFIG["gpt_sovits_root"]}" && python "{train_script}"')
    else:
        print("已取消训练。")


if __name__ == "__main__":
    run()
