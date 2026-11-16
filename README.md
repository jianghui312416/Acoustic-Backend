# 声场可视化软件后端

这是按 `D:\Acoustic_Field_GeoVis` 当前前端调用链整理的最小可运行后端。默认监听 `0.0.0.0:8888`。

## 启动

```powershell
cd D:\声场可视化软件后端
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\start_backend.ps1
```

启动后可访问 `http://127.0.0.1:8888/docs`。前端的 `VITE_PORTAL_API_URL` 应设置为 `http://127.0.0.1:8888/`。

## 配置

- `config/data_config.json`：环境数据、任务目录、门户参数与噪声模型。
- `config/bellhop_config.json`、`ram_config.json`、`kraken_config.json`：声学模型。
- `config/global_*_noise*.json`：前端环境噪声接口按需构建源产品时使用。
- AISStream 密钥使用环境变量 `AISSTREAM_API_KEY`，或写入 `config/aisstream_api_key.txt`。

水体温度、盐度和海冰厚度由后端按前端所选时间从 HYCOM 官方接口读取，自动选择历史归档、当前分析或预报产品；无需前端选择数据源。海面由请求中的 `seaState` 生成，不再依赖本地波浪文件。天气、非冰区噪声和 AIS 仍由后端按需请求或计算；冰区环境噪声接口已移除。其他环境 NetCDF/AIS 原始数据未复制进本目录；配置已指向同盘的 `D:\JD功能开发\HYdata` 与 `local_datafile`。任务结果写入本目录下的 `jobs`。

## 保留的前端能力

- Bellhop、RAM、Kraken 的二维/三维传播损失及 Bellhop 声线；
- 海冰场景；
- 天气、天气驱动噪声、手动环境噪声、实时 AIS；
- HYCOM 历史/当前/预报时间、航线和区域查询；
- 声场任务工作区、阵列静态信道、通信链路与实时 WebSocket。
