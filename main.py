"""
PersonaChat Backend - FastAPI Server
提供 DeepSeek 对话 API、edge-tts 语音合成、可选整合 GPT-SoVITS 本地推理
"""
import json
import asyncio
import tempfile
import uuid
import os
import sys
import shutil
import subprocess
import time
import atexit
import signal
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

# edge-tts
import edge_tts

# ─────────────────────────────────────────────
# 翻译函数（中译日）
# ─────────────────────────────────────────────
async def translate_zh_to_ja(text: str, api_key: str) -> str:
    """使用 DeepSeek API 将中文翻译成日语"""
    if not text.strip():
        return text

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "你是一个专业的日语翻译。请将用户给出的中文句子翻译成自然流畅的日语，不要添加任何标记或解释，直接输出翻译结果。"},
                        {"role": "user", "content": text}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.3,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return text

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
TEMP_DIR = Path(os.getenv("PERSONACHAT_TEMP_DIR", str(BASE_DIR / "temp")))
TEMP_DIR.mkdir(exist_ok=True)

GPTSOVITS_CONFIG_PATH = BASE_DIR / "gpt_sovits_config.json"
_gptsovits_config: Optional[dict] = None
_gptsovits_process: Optional[subprocess.Popen] = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cors_origins() -> list[str]:
    configured = os.getenv("PERSONACHAT_CORS_ORIGINS", "*").strip()
    if not configured:
        return ["*"]
    return [item.strip() for item in configured.split(",") if item.strip()]

# ─────────────────────────────────────────────
# Pydantic 模型
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    character_id: str
    api_key: str
    history: list[dict] = []

class ChatMessage(BaseModel):
    role: str
    content: str

class TTSRequest(BaseModel):
    text: str
    voice: str = "zh-CN-Xiaoxiao"
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"

class GptsovitsRequest(BaseModel):
    text: str
    ref_audio_path: str = ""
    ref_audio_base64: str = ""
    prompt_text: str = ""
    prompt_language: str = "all_zh"
    text_language: str = "all_zh"

class GptsovitsReloadRequest(BaseModel):
    gpt_weights: str = ""
    sovits_weights: str = ""

class Character(BaseModel):
    id: str
    name: str
    name_en: str
    gender: str
    age: str
    personality: str
    style: str
    greeting: str
    avatar_color: str
    system_prompt: str
    tts_voice: str

# ─────────────────────────────────────────────
# 预置角色数据
# ─────────────────────────────────────────────
def build_system_prompt(char: Character) -> str:
    return (
        f"你正在扮演一个名叫 {char.name} ({char.name_en}) 的角色。\n"
        f"性别: {char.gender}，年龄: {char.age}岁。\n"
        f"性格: {char.personality}。\n"
        f"说话风格: {char.style}\n\n"
        f"请始终以这个角色的身份和口吻回复，语气、措辞都要符合角色设定。不要打破第四面墙。\n"
        f"【重要】请用日语回答。回复格式如下：\n"
        f"|JA| 日语原文（用于语音合成）\n"
        f"|ZH| 中文翻译（用于屏幕显示）"
    )

def get_builtin_characters() -> list[Character]:
    return [
        Character(
            id="xiaoyuruyu",
            name="萧容鱼",
            name_en="Xiao Yu Ru Yu",
            gender="女",
            age="18",
            personality="傲娇可人、独立坚强、甜美活泼、自信骄傲，有一对可爱的梨涡，爱扎高马尾，是东大校花。性格倔强有原则，对感情既期待也有疑虑，但内心柔软善良。",
            style="甜美活泼，说话娇俏灵动，常用语气词如\"哼\"\"呢\"\"啦\"，生气时语气激烈但不失分寸，有时傲娇别扭。",
            greeting="哼，终于舍得来找我啦？我可是东海大学校花萧容鱼，有什么想聊的尽管说，不过别太无聊哦。",
            avatar_color="#FF69B4",
            tts_voice="zh-CN-XiaoxiaoNeural",
            system_prompt=""
        ),
    ]

# ─────────────────────────────────────────────
# FastAPI 应用
# ─────────────────────────────────────────────
app = FastAPI(title="PersonaChat API", version="1.0.0")

_ALLOW_LOCAL_VIDEO = _env_flag("PERSONACHAT_ALLOW_LOCAL_VIDEO", default=False)
_CORS_ORIGINS = _cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials="*" not in _CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# 路由
# ─────────────────────────────────────────────
@app.get("/api/characters")
async def get_characters():
    """获取所有可用角色列表"""
    chars = get_builtin_characters()
    return {"characters": chars}


@app.get("/")
async def root():
    return {"name": "PersonaChat API", "status": "ok", "docs": "/docs"}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """发送消息给 DeepSeek AI，返回角色扮演的回复"""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    if not req.api_key.strip():
        raise HTTPException(status_code=401, detail="请先配置 DeepSeek API Key")

    # 查找角色
    builtin = {c.id: c for c in get_builtin_characters()}
    char = builtin.get(req.character_id)

    if char is None:
        raise HTTPException(status_code=404, detail=f"角色 {req.character_id} 不存在")

    system_prompt = build_system_prompt(char)

    # 构建消息历史
    messages = [{"role": "system", "content": system_prompt}]
    for h in req.history[-20:]:  # 限制历史长度
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": req.message})

    # 调用 DeepSeek API
    headers = {
        "Authorization": f"Bearer {req.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.9,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=payload,
            )
            if resp.status_code != 200:
                error_detail = resp.json().get("error", {})
                msg = error_detail.get("message", resp.text)
                raise HTTPException(status_code=resp.status_code, detail=f"DeepSeek API 错误: {msg}")
            data = resp.json()
            raw_reply = data["choices"][0]["message"]["content"]

            # 解析回复格式：|JA| 日语原文 和 |ZH| 中文翻译
            import re
            ja_match = re.search(r'\|JA\|\s*(.+?)(?:\||\Z)', raw_reply, re.DOTALL)
            zh_match = re.search(r'\|ZH\|\s*(.+?)(?:\||\Z)', raw_reply, re.DOTALL)

            zh_text = zh_match.group(1).strip() if zh_match else raw_reply.strip()

            # 自动翻译中文回复为日语
            ja_text = await translate_zh_to_ja(zh_text, req.api_key)

            return {"reply": zh_text, "reply_ja": ja_text, "error": None}
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="请求超时，请稍后重试")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """使用 edge-tts 将文字转为语音文件"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    # 限制文本长度
    text = req.text[:1000]

    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = TEMP_DIR / filename

    try:
        communicate = edge_tts.Communicate(
            text,
            req.voice,
            rate=req.rate,
            pitch=req.pitch,
            volume=req.volume,
        )
        await communicate.save(str(filepath))
        return FileResponse(
            filepath,
            media_type="audio/mp3",
            filename=filename,
            background=BackgroundTask(_cleanup_file, str(filepath))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 错误: {str(e)}")

@app.get("/api/tts/voices")
async def get_voices():
    """获取 edge-tts 可用音色列表"""
    try:
        voices = await edge_tts.list_voices()
        # 筛选中文音色
        zh_voices = [v for v in voices if v["Locale"].startswith("zh-")]
        return {"voices": zh_voices}
    except Exception:
        return {"voices": []}

@app.post("/api/gptsovits")
async def gptsovits_tts(req: GptsovitsRequest):
    """调用 GPT-SoVITS 进行语音合成（自动管理子进程 + HTTP 转发）"""
    import base64

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    text = req.text[:1000]

    cfg = _load_gptsovits_config()
    if not cfg.get("enabled"):
        raise HTTPException(status_code=400, detail="GPT-SoVITS 未启用，请在 backend/gpt_sovits_config.json 中设置 enabled=true")

    default = cfg.get("default_model", {})

    # ── 预检验：检查配置完整性 ──
    gpt_weights = default.get("gpt_weights", "")
    sovits_weights = default.get("sovits_weights", "")
    ref_audio_path = req.ref_audio_path or default.get("ref_audio_path", "")
    ref_text = req.prompt_text or default.get("ref_text", "")
    prompt_lang = req.prompt_language or default.get("prompt_lang", "zh")
    text_lang = req.text_language or default.get("text_lang", "zh")

    missing = []
    import os as _os
    print(f"[DEBUG ref_audio_path] value={ref_audio_path!r}, exists={_os.path.exists(ref_audio_path) if ref_audio_path else False}")
    if not gpt_weights or not _os.path.exists(gpt_weights):
        missing.append(f"GPT 权重不存在: {gpt_weights}")
    if not sovits_weights or not _os.path.exists(sovits_weights):
        missing.append(f"SoVITS 权重不存在: {sovits_weights}")
    if not ref_audio_path or not _os.path.exists(ref_audio_path):
        missing.append(f"参考音频不存在: {ref_audio_path}")
    if not ref_text or not ref_text.strip():
        missing.append("参考文本未填写")

    if missing:
        raise HTTPException(
            status_code=400,
            detail="GPT-SoVITS 配置不完整:\n" + "\n".join(f"  • {m}" for m in missing)
        )

    # 确保子进程在跑
    if not await _ensure_gptsovits_running():
        raise HTTPException(status_code=503, detail="GPT-SoVITS 服务启动失败，请检查 backend 控制台日志")

    # 设置默认模型（子进程第一次启动后只用设一次）
    await _set_gptsovits_default_model(gpt_weights, sovits_weights)

    # 构建 api_v2 的请求体
    lang_map = {
        "zh": "zh", "中文": "zh", "chinese": "zh",
        "en": "en", "英文": "en", "english": "en",
        "ja": "ja", "日文": "ja", "japanese": "ja", "jp": "ja",
        "ko": "ko", "韩文": "ko", "korean": "ko",
        "yue": "yue", "粤语": "yue", "cantonese": "yue",
        "auto": "auto", "auto_zh": "auto", "auto_en": "auto",
        "auto_ja": "auto", "auto_ko": "auto", "auto_yue": "auto_yue",
        "all_zh": "all_zh", "all_en": "all_zh", "all_ja": "all_ja",
        "all_ko": "all_ko", "all_yue": "all_yue",
        "zh_en": "zh", "中英混合": "zh", "ja_en": "ja", "日英混合": "ja",
    }
    text_lang = lang_map.get(text_lang, text_lang)
    prompt_lang = lang_map.get(prompt_lang, prompt_lang)

    payload = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_text": ref_text,
        "prompt_lang": prompt_lang,
        "media_type": "wav",
        "streaming_mode": False,
    }
    print(f"[DEBUG GPT-SoVITS] ref_audio_path={ref_audio_path!r}, text_lang={text_lang}, prompt_lang={prompt_lang}")

    host = cfg.get("host", "127.0.0.1")
    port = cfg.get("port", 9880)
    url = f"http://{host}:{port}/tts"

    filename = f"{uuid.uuid4().hex}.wav"
    filepath = TEMP_DIR / filename

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            print(f"[DEBUG GPT-SoVITS] status={resp.status_code} body={resp.text[:500]}")
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"GPT-SoVITS 返回错误: {resp.status_code} {resp.text[:500]}",
                )

            ctype = resp.headers.get("content-type", "")
            if "application/json" in ctype:
                # api_v2 出错时也可能返回 JSON
                try:
                    err = resp.json()
                except Exception:
                    err = {"raw": resp.text[:500]}
                raise HTTPException(status_code=502, detail=f"GPT-SoVITS 出错: {err}")

            # 成功：直接把 wav 流写入文件
            with open(filepath, "wb") as f:
                f.write(resp.content)

        return FileResponse(
            filepath,
            media_type="audio/wav",
            filename=filename,
            background=BackgroundTask(_cleanup_file, str(filepath)),
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="GPT-SoVITS 请求超时（>120s）")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT-SoVITS TTS 错误: {str(e)}")

@app.get("/api/video")
async def serve_video(file_path: str):
    """提供本地视频文件的访问"""
    from fastapi import Header
    try:
        if not _ALLOW_LOCAL_VIDEO:
            raise HTTPException(status_code=403, detail="Local video serving is disabled in deployed environments")
        video_path = Path(file_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")
        if not video_path.is_file():
            raise HTTPException(status_code=400, detail="路径不是文件")
        mime_types = {
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska",
        }
        ext = video_path.suffix.lower()
        media_type = mime_types.get(ext, "video/mp4")
        return FileResponse(
            video_path,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/gptsovits/reload")
async def gptsovits_reload(req: GptsovitsReloadRequest):
    """手动切换 GPT/SoVITS 权重（不传则用配置里的默认）"""
    cfg = _load_gptsovits_config()
    if not cfg.get("enabled"):
        raise HTTPException(status_code=400, detail="GPT-SoVITS 未启用")

    if not await _ensure_gptsovits_running():
        raise HTTPException(status_code=503, detail="GPT-SoVITS 服务不可用")

    default = cfg.get("default_model", {})
    gpt = req.gpt_weights or default.get("gpt_weights", "")
    sovits = req.sovits_weights or default.get("sovits_weights", "")

    await _set_gptsovits_default_model(gpt, sovits)
    return {
        "ok": True,
        "gpt_weights": gpt,
        "sovits_weights": sovits,
    }

async def _test_gptsovits_tts(host: str, port: int, timeout: float = 15.0) -> dict:
    """真正发送一个短文本给 GPT-SoVITS，测试它是否能正常合成"""
    cfg = _load_gptsovits_config()
    default = cfg.get("default_model", {})
    ref_audio = default.get("ref_audio_path", "")
    ref_text = default.get("ref_text", "")
    p_lang = default.get("prompt_lang", "zh")
    t_lang = default.get("text_lang", "zh")
    lang_map = {
        "zh": "zh", "中文": "zh", "chinese": "zh",
        "en": "en", "英文": "en", "english": "en",
        "ja": "ja", "日文": "ja", "japanese": "ja", "jp": "ja",
        "ko": "ko", "韩文": "ko", "korean": "ko",
        "yue": "yue", "粤语": "yue", "cantonese": "yue",
        "auto": "auto", "auto_zh": "auto", "auto_en": "auto",
        "auto_ja": "auto", "auto_ko": "auto", "auto_yue": "auto_yue",
        "all_zh": "all_zh", "all_en": "all_zh", "all_ja": "all_ja",
        "all_ko": "all_ko", "all_yue": "all_yue",
        "zh_en": "zh", "中英混合": "zh", "ja_en": "ja", "日英混合": "ja",
    }
    p_lang_m = lang_map.get(p_lang, p_lang)
    t_lang_m = lang_map.get(t_lang, t_lang)

    payload = {
        "text": "こんにちは",
        "text_lang": t_lang_m,
        "ref_audio_path": ref_audio,
        "prompt_text": ref_text,
        "prompt_lang": p_lang_m,
        "media_type": "wav",
        "streaming_mode": False,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"http://{host}:{port}/tts", json=payload)
            if resp.status_code == 200 and len(resp.content) > 1000:
                return {"ok": True, "size": len(resp.content)}
            else:
                return {"ok": False, "reason": f"HTTP {resp.status_code}, size={len(resp.content)}"}
    except httpx.TimeoutException:
        return {"ok": False, "reason": f"超时（>{timeout}s）—— GPT-SoVITS 可能卡住了，检查控制台日志"}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


@app.get("/api/health")
async def health():
    cfg = _load_gptsovits_config()
    host = cfg.get("host", "127.0.0.1")
    port = cfg.get("port", 9880)
    gptsovits_port_open = _is_gptsovits_port_open(host, port)
    enabled = cfg.get("enabled", False)

    default = cfg.get("default_model", {})
    gpt_path = default.get("gpt_weights", "")
    sovits_path = default.get("sovits_weights", "")
    ref_audio_path = default.get("ref_audio_path", "")
    ref_text = default.get("ref_text", "")

    gpt_exists = bool(gpt_path) and Path(gpt_path).exists()
    sovits_exists = bool(sovits_path) and Path(sovits_path).exists()
    ref_exists = bool(ref_audio_path) and Path(ref_audio_path).exists()
    print(f"[HEALTH CHECK] ref_audio_path={ref_audio_path!r}, exists={ref_exists}")
    ref_text_ok = bool(ref_text.strip()) and ref_text.strip() not in ("请填入参考音频对应的中文文本", "")

    prompt_lang = default.get("prompt_lang", "zh")
    text_lang = default.get("text_lang", "zh")
    auto_prefixes = ("auto_", "all_")
    lang_compatible = (
        prompt_lang == text_lang
        or text_lang.startswith("auto_")
        or prompt_lang.startswith("auto_")
        or text_lang in auto_prefixes
        or prompt_lang in auto_prefixes
    )

    warnings = []
    if enabled and gptsovits_port_open:
        if not gpt_exists:
            warnings.append(f"❌ GPT 权重文件不存在: {gpt_path}")
        if not sovits_exists:
            warnings.append(f"❌ SoVITS 权重文件不存在: {sovits_path}")
        if not ref_exists:
            warnings.append(f"❌ 参考音频文件不存在: {ref_audio_path}")
        if not ref_text_ok:
            warnings.append(f"⚠️ 参考文本未填写或为占位符")
        if not lang_compatible:
            warnings.append(
                f"⚠️ prompt_lang='{prompt_lang}' 与 text_lang='{text_lang}' 不一致，"
                f"跨语言合成质量会下降，建议设为相同语言。"
            )

    # 真正测试 TTS 合成（仅在端口开放时）
    tts_test: Optional[dict] = None
    if enabled and gptsovits_port_open:
        tts_test = await _test_gptsovits_tts(host, port, timeout=15.0)
        if not tts_test["ok"]:
            warnings.append(f"❌ GPT-SoVITS TTS 测试失败: {tts_test['reason']}")

    return {
        "status": "ok",
        "gpt_sovits_enabled": enabled,
        "gpt_sovits_url": f"http://{host}:{port}" if enabled else None,
        "gpt_sovits_port_open": gptsovits_port_open,
        "gpt_sovits_tts_test": tts_test,
        "gpt_sovits_pid": _gptsovits_process.pid if (_gptsovits_process and _gptsovits_process.poll() is None) else None,
        "model_check": {
            "gpt_weights": gpt_path,
            "gpt_exists": gpt_exists,
            "sovits_weights": sovits_path,
            "sovits_exists": sovits_exists,
            "ref_audio_path": ref_audio_path,
            "ref_audio_exists": ref_exists,
            "ref_text": ref_text,
            "ref_text_ok": ref_text_ok,
            "prompt_lang": prompt_lang,
            "text_lang": text_lang,
            "lang_compatible": lang_compatible,
        },
        "warnings": warnings,
    }

# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────
async def _cleanup_file(filepath: str):
    """后台清理临时音频文件（延迟删除，浏览器有足够时间下载）"""
    await asyncio.sleep(300)
    try:
        Path(filepath).unlink(missing_ok=True)
    except Exception:
        pass

# ─────────────────────────────────────────────
# GPT-SoVITS 子进程管理
# ─────────────────────────────────────────────
def _load_gptsovits_config() -> dict:
    global _gptsovits_config
    if _gptsovits_config is not None:
        return _gptsovits_config
    if not GPTSOVITS_CONFIG_PATH.exists():
        _gptsovits_config = {"enabled": False}
        return _gptsovits_config
    try:
        with open(GPTSOVITS_CONFIG_PATH, "r", encoding="utf-8") as f:
            _gptsovits_config = json.load(f)
    except Exception as e:
        print(f"[GPT-SoVITS] 读取配置失败: {e}")
        _gptsovits_config = {"enabled": False}
    env_enabled = os.getenv("PERSONACHAT_ENABLE_GPTSOVITS")
    if env_enabled is not None:
        _gptsovits_config["enabled"] = _env_flag("PERSONACHAT_ENABLE_GPTSOVITS", default=False)
    return _gptsovits_config

def _is_gptsovits_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

async def _wait_gptsovits_ready(host: str, port: int, timeout: float) -> bool:
    """轮询直到 api_v2 上线（或超时）"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_gptsovits_port_open(host, port, timeout=0.5):
            return True
        await asyncio.sleep(1.0)
    return False

async def _ensure_gptsovits_running() -> bool:
    """确保 GPT-SoVITS api_v2 子进程在跑；如未跑则启动"""
    global _gptsovits_process

    cfg = _load_gptsovits_config()
    host = cfg.get("host", "127.0.0.1")
    port = cfg.get("port", 9880)

    # 已经在跑
    if _gptsovits_process is not None and _gptsovits_process.poll() is None:
        if _is_gptsovits_port_open(host, port):
            return True

    # 端口已经被别的进程占了（用户在外部启动了 GPT-SoVITS）
    if _is_gptsovits_port_open(host, port):
        print(f"[GPT-SoVITS] 检测到外部进程已在 {host}:{port}，直接复用")
        return True

    if not cfg.get("auto_start", True):
        print("[GPT-SoVITS] auto_start=false 且无外部进程，跳过启动")
        return False

    work_dir = cfg.get("work_dir", "")
    python_exe = cfg.get("python_exe", "")
    api_script = cfg.get("api_script", "api_v2.py")
    tts_config = cfg.get("tts_config", "GPT_SoVITS/configs/tts_infer.yaml")

    if not work_dir or not python_exe:
        print("[GPT-SoVITS] work_dir 或 python_exe 未配置")
        return False
    if not Path(python_exe).exists():
        print(f"[GPT-SoVITS] Python 不存在: {python_exe}")
        return False
    if not (Path(work_dir) / api_script).exists():
        print(f"[GPT-SoVITS] API 脚本不存在: {work_dir}\\{api_script}")
        return False

    cmd = [
        python_exe, "-I", api_script,
        "-a", host,
        "-p", str(port),
        "-c", tts_config,
    ]
    print(f"[GPT-SoVITS] 启动子进程: {' '.join(cmd)}")
    print(f"[GPT-SoVITS] 工作目录: {work_dir}")

    # Windows 上不要隐藏窗口：api_v2 启动慢且可能报错，要能看到
    # 注意：CREATE_NEW_CONSOLE + PIPE 在 Windows 上行为古怪（PIPE 会断），
    # 所以我们不开新窗口，直接继承父进程 stdout（子进程会输出到我们的 console）
    creationflags = 0
    if sys.platform == "win32":
        # CREATE_NEW_CONSOLE 强制开新窗口
        creationflags = 0  # 关闭新窗口，直接复用我们的 console
        # 如果你想看 api_v2 自己独立的黑窗口，把 0 改成 subprocess.CREATE_NEW_CONSOLE
        # 并且下面 stdout= 也别改成 None。但那种情况下 Windows 会把输出路由到 PIPE 失败。

    try:
        _gptsovits_process = subprocess.Popen(
            cmd,
            cwd=work_dir,
            stdout=None,  # 继承父进程 stdout（子进程输出会直接显示在我们的 console）
            stderr=None,  # 继承父进程 stderr
            creationflags=creationflags,
        )
    except Exception as e:
        print(f"[GPT-SoVITS] 启动失败: {e}")
        return False

    print(f"[GPT-SoVITS] 子进程 PID: {_gptsovits_process.pid}")

    timeout = float(cfg.get("startup_timeout_sec", 90))
    print(f"[GPT-SoVITS] 等待子进程就绪（最多 {timeout:.0f}s）...")
    ok = await _wait_gptsovits_ready(host, port, timeout)
    if not ok:
        print("[GPT-SoVITS] 子进程启动超时")
        try:
            _gptsovits_process.terminate()
        except Exception:
            pass
        _gptsovits_process = None
        return False

    print(f"[GPT-SoVITS] 已就绪: http://{host}:{port}")
    return True

async def _set_gptsovits_default_model(gpt_weights: str, sovits_weights: str) -> None:
    """启动后让子进程加载默认模型权重"""
    cfg = _load_gptsovits_config()
    host = cfg.get("host", "127.0.0.1")
    port = cfg.get("port", 9880)
    base = f"http://{host}:{port}"

    async with httpx.AsyncClient(timeout=60.0) as client:
        if gpt_weights:
            try:
                r = await client.get(f"{base}/set_gpt_weights", params={"weights_path": gpt_weights})
                print(f"[GPT-SoVITS] set_gpt_weights: {r.status_code} {r.text[:200]}")
            except Exception as e:
                print(f"[GPT-SoVITS] 设置 GPT 权重失败: {e}")
        if sovits_weights:
            try:
                r = await client.get(f"{base}/set_sovits_weights", params={"weights_path": sovits_weights})
                print(f"[GPT-SoVITS] set_sovits_weights: {r.status_code} {r.text[:200]}")
            except Exception as e:
                print(f"[GPT-SoVITS] 设置 SoVITS 权重失败: {e}")

def _shutdown_gptsovits():
    global _gptsovits_process
    if _gptsovits_process is None:
        return
    if _gptsovits_process.poll() is not None:
        return
    print("[GPT-SoVITS] 关闭子进程...")
    try:
        if sys.platform == "win32":
            _gptsovits_process.terminate()
        else:
            _gptsovits_process.send_signal(signal.SIGTERM)
        try:
            _gptsovits_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _gptsovits_process.kill()
    except Exception as e:
        print(f"[GPT-SoVITS] 关闭出错: {e}")
    _gptsovits_process = None

atexit.register(_shutdown_gptsovits)

async def _background_warmup():
    """uvicorn 启动后，后台异步拉起 api_v2 + 切模型"""
    cfg = _load_gptsovits_config()
    print("[GPT-SoVITS] 后台预热：拉起 api_v2 子进程...")
    ok = await _ensure_gptsovits_running()
    if not ok:
        print("[GPT-SoVITS] 后台预热失败：api_v2 未起来（合成时会再试）")
        return

    default = cfg.get("default_model", {})
    gpt = default.get("gpt_weights", "")
    sovits = default.get("sovits_weights", "")
    print(f"[GPT-SoVITS] 加载模型: gpt={gpt.split(chr(92))[-1]}, sovits={sovits.split(chr(92))[-1]}")
    try:
        await _set_gptsovits_default_model(gpt, sovits)
        print("[GPT-SoVITS] 后台预热完成 ✓")
    except Exception as e:
        print(f"[GPT-SoVITS] 切模型失败: {e}")

# ─────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    cfg = _load_gptsovits_config()
    if cfg.get("enabled"):
        print(f"[GPT-SoVITS] 已启用，模型: {cfg.get('default_model', {}).get('sovits_weights', '?')}")
        if cfg.get("auto_start", True):
            # 后台异步启动 GPT-SoVITS + 切模型，不阻塞 uvicorn 启动
            import threading

            def _bg_warmup():
                try:
                    asyncio.run(_background_warmup())
                except Exception as e:
                    print(f"[GPT-SoVITS] 后台预热失败: {e}")

            t = threading.Thread(target=_bg_warmup, daemon=True)
            t.start()
    # 注意：禁用 --reload。reload 模式下子进程会被 reloader 视为 worker，
    # 子进程里再 Popen api_v2 会被反复杀掉。
    uvicorn.run(
        "main:app",
        host=os.getenv("PERSONACHAT_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", os.getenv("PERSONACHAT_PORT", "8000"))),
    )
