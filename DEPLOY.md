# PersonaChat Backend Deploy

这个目录是从原 `cursor/backend` 单独整理出来的可部署版本，核心聊天逻辑保持原样，只补了上线需要的最小配置。

## 本地启动

```bash
cd promo-backend
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

接口默认地址：

`http://localhost:8000`

文档地址：

`http://localhost:8000/docs`

## 线上部署建议

推荐把它部署到支持 Python 常驻服务的平台，例如：

- Render
- Railway
- 云服务器 / Docker

不建议部署到纯前端平台，因为这是长期运行的 FastAPI 服务。

## 关键环境变量

- `PERSONACHAT_CORS_ORIGINS`
  你的前端域名，多个域名可用逗号分隔
- `PERSONACHAT_ENABLE_GPTSOVITS`
  默认 `false`
- `PERSONACHAT_ALLOW_LOCAL_VIDEO`
  默认 `false`
- `PORT`
  平台自动注入时无需手填

## 关于 GPT-SoVITS

云端默认关闭了 GPT-SoVITS，因为它依赖：

- 本地模型路径
- 本地 Python 运行时
- GPU / CUDA 环境

如果以后你真的要在线启用它，建议单独上 GPU 服务器，再把：

`PERSONACHAT_ENABLE_GPTSOVITS=true`

并补齐 `gpt_sovits_config.json` 里的真实路径。

## 关于视频背景

`/api/video` 现在默认关闭，因为它原本是读取本机磁盘文件。线上环境没有你电脑里的那些路径，所以继续开放反而会报错。

如果以后你要保留视频背景，更适合把视频放到：

- 前端 `public/`
- 对象存储
- CDN

## 与前端联动

前端部署后，把前端环境变量设成：

`VITE_API_BASE=https://你的后端域名/api`

这样 `promo-site` 就会请求线上后端，而不是本地 `/api`。
