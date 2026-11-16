# api_endpoints → C++ 转换设计与映射表

> 范围：仅转换部分接口（分批进行）。技术栈：C++17 + cpp-httplib + nlohmann/json + CMake。
> 源目录：`D:\JD功能开发\backworker\api_endpoints`
> 输出目录：`D:\JD转换C`

---

## 1. 结论先行

原 `api_endpoints` **不是**独立业务实现，而是 FastAPI 的薄封装层：

- `common.py`：统一读取门户请求、统一包装响应、统一异常处理、创建单接口路由。
- `registry.py`：`Plugin → SN` 固定对应表（43 个 Plugin）。
- `dispatcher.py`：统一入口 `POST /`，按 `Data.SN` + `Data.Units[0].Plugin` 校验后分派。
- 各 `*.py`：只做「字段白名单校验 → 组装参数 → 调 `backend_worker` / `services` → 返回业务 JSON」。
- 真正的重逻辑在 `backworker/backend_worker.py`（187KB）、`backworker/services/*`、`bellhop/kraken/ram` 外部程序与 numpy 数值计算中。

因此 C++ 化的**合理切分**是：

1. **P0 协议层（必做）**：HTTP 入口、JSON 解析、Plugin/SN 校验、请求/响应包装、错误码、字段校验工具、路由注册。
2. **P1 查询类**：`GetEnvDataSourceList`、`FetchTimeToList`、`GetDataTypeMetaInfo`。
3. **P2 环境数据类**：`GetEnvDataStation/Route/Region`。
4. **P3 数值分析类**：声速/跃层/声道/波导/混响等。
5. **P4 声传播类**：2D/3D TL/Ray（依赖 bellhop.exe / kraken.exe / ram.exe + NetCDF）。
6. **P5 大规模编排类**：环境噪声、天气预报、AIS。

> 建议先做 **P0 + P1** 作为可编译、可联调的试点，再逐批推进。完整清单见第 5 节与 `endpoint_mapping.csv`。

---

## 2. 现有 Python 架构

### 2.1 两个 HTTP 入口

1. **统一入口**：`POST /`
   - 由 `dispatcher.create_router()` 注册。
   - 从 `Data.SN`、`Data.Units[0].Plugin` 校验 `registry.PORTAL_PLUGIN_SN_MAP`。
   - 校验通过后调用对应 `handle(payload)`。
2. **每插件独立路由**：如 `POST /Analyse2DTLRD`
   - 由各文件 `create_router(openapi_extra)` 注册。
   - **不做 SN 校验**，直接进入 `handle(payload)`。
3. **特殊直连路由**：
   - `POST /waveguide/analyse`（`waveguide_dispersion.py`，无 Plugin/SN）。
   - `POST /AnalyseReverberationWaveform`（复用 `AnalyseReverberation` 的 Plugin/SN）。

### 2.2 统一处理链（`common.handle_portal_payload`）

```
payload
  → core._reload_config_if_changed()
  → core._portal_unit_request(payload, plugin_name)        # 取 Units[].Request 或平铺请求
  → build_response(request_payload, core, meta)            # 各接口业务
      ├─ _reject_unsupported_request_fields(...)           # 字段白名单
      ├─ _portal_*_params / _api_float / _api_int / ...    # 参数归一化
      ├─ _run_portal_job / services.xxx                    # 计算
      └─ _build_*_response                                 # 业务 JSON
  → 若 job_id 存在：写 jobs/<job_id>/response_<plugin>.json
  → core._wrap_portal_response(...)                        # 包装统一响应
  → core._api_json_response(...)                           # JSON + Content-Type
异常：
  PortalRequestError → 400（按 meta 决定 wrapped / flat）
  Exception          → 500
```

### 2.3 请求格式

**统一包装格式（wrapped）**

```json
{
  "Technical Support": "XWOcean",
  "Data": {
    "SN": 16,
    "TaskId": "",
    "SelfOrder": 0,
    "Units": [
      {
        "Plugin": "Analyse2DTLRD",
        "Request": { "coordX": 116.0, "coordY": 16.0, "srcDepth": 50.0, "time": "2025-01-04 12:00:00" }
      }
    ]
  }
}
```

**平铺兼容格式（flat）**

```json
{ "coordX": 116.0, "coordY": 16.0, "srcDepth": 50.0, "time": "2025-01-04 12:00:00" }
```

判定规则：payload 中存在 `Data` 对象且含 `Units` 数组 → wrapped；否则 → flat。
`Data.Units` 在统一入口必须且只能有 1 个元素；单接口路由允许取第一个匹配 Plugin 的单元。

### 2.4 响应格式

**wrapped 成功**

```json
{
  "Technical Support": "XWOcean",
  "Data": {
    "SN": 16,
    "TaskId": "abcdef123456",
    "SelfOrder": 0,
    "Units": [
      { "Plugin": "Analyse2DTLRD", "Response": { "...": "业务结果" }, "ResultCode": 0 }
    ]
  }
}
```

**flat 成功**：业务结果对象直接展开，并追加 `"ResultCode": 0`。
若业务结果是数组/标量，则包成 `{ "Response": <value>, "ResultCode": 0 }`。

**失败**

```json
{ "ok": false, "message": "错误说明", "ResultCode": 1 }
```

wrapped 失败时 `Response` 内为上述对象，HTTP 状态 400/500。

### 2.5 关键常量

| 常量 | 值 | 位置 |
|---|---|---|
| `NM_TO_KM` | 1.852 | backend_worker.py:153 |
| `DEFAULT_RESPONSE_ROW` | 100 | backend_worker.py:154 |
| `DEFAULT_RESPONSE_COL` | 300 | backend_worker.py:155 |
| `PORTAL_DECIMAL_PLACES` | 6 | backend_worker.py:156 |
| `JOBS_DIR` | 配置 `JOBS_DIR` | backend_worker.py:122 |

---

## 3. C++ 分层设计

```
┌──────────────────────────────────────────────┐
│ main.cpp / http_server                       │  httplib::Server
├──────────────────────────────────────────────┤
│ dispatcher                                   │  统一入口 + SN/Plugin 校验
│   resolve_portal_handler(payload)            │
├──────────────────────────────────────────────┤
│ registry                                     │  Plugin → {SN, Handler}
├──────────────────────────────────────────────┤
│ portal_protocol (对应 common.py)             │  请求/响应包装、错误、字段校验
│   - handle_portal_payload                    │
│   - portal_unit_request                      │
│   - wrap_portal_response                     │
│   - portal_failure                           │
│   - reject_unsupported_request_fields        │
│   - api_float / api_int / api_strict_bool    │
├──────────────────────────────────────────────┤
│ endpoints/<plugin>.cpp                       │  各接口：字段校验 + 业务
├──────────────────────────────────────────────┤
│ core (对应 backend_worker 公共函数)          │  参数归一化、环境数据、任务执行
│ services (对应 services/*)                   │  数值算法
│ external (exe / NetCDF / DB / HTTP)          │  外部依赖适配
└──────────────────────────────────────────────┘
```

### 3.1 建议目录

```
D:\JD转换C\
  CMakeLists.txt
  README.md
  API_CPP_DESIGN.md
  endpoint_mapping.csv
  include\jd\
    json.h
    portal_error.h
    portal_protocol.h
    registry.h
    dispatcher.h
    config.h
    jobs.h
    http_server.h
    core\
      api_params.h
      env_data.h
      acoustic_job.h
    endpoints\
      get_dataset_list.h
      get_dataset_time_list.h
      get_data_type_meta_list.h
      get_env_data_station.h
      get_env_data_route.h
      get_env_data_region.h
      ...
  src\
    main.cpp
    http_server.cpp
    portal_protocol.cpp
    registry.cpp
    dispatcher.cpp
    config.cpp
    jobs.cpp
    core\
      api_params.cpp
      env_data.cpp
      acoustic_job.cpp
    endpoints\
      get_dataset_list.cpp
      get_dataset_time_list.cpp
      get_data_type_meta_list.cpp
      ...
  third_party\        # 可选；也可用 CMake FetchContent 拉取
  tests\
    test_protocol.cpp
    test_registry.cpp
```

### 3.2 核心类型

```cpp
namespace jd {

using Json = nlohmann::json;

class PortalRequestError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct PortalMeta {
    bool wrapped = false;
    std::string technical_support = "XWOcean";
    Json sn = "";
    Json task_id = "";
    Json self_order = 0;
    std::string plugin;
};

struct HttpResult {
    int status = 200;
    Json body;
    std::string content_type = "application/json; charset=utf-8";
};

using Handler = std::function<HttpResult(const Json& payload)>;
using Business = std::function<std::pair<Json, std::optional<std::string>>(
    const Json& request_payload, const PortalMeta& meta)>;

} // namespace jd
```

### 3.3 协议函数映射

| Python | C++ 建议签名 |
|---|---|
| `_request_payload(request)` | `Json parse_request_body(const std::string& body, const std::string& content_type)` |
| `_portal_unit_request(payload, plugin)` | `std::pair<Json, PortalMeta> portal_unit_request(const Json&, const std::string&)` |
| `_wrap_portal_response(...)` | `Json wrap_portal_response(const Json&, const PortalMeta&, const std::string&, const std::optional<std::string>&, int result_code = 0)` |
| `_flat_portal_response(...)` | `Json flat_portal_response(const Json&, int result_code = 0)` |
| `_portal_failure(...)` | `HttpResult portal_failure(const std::exception&, int status, const std::optional<PortalMeta>&, const std::string&)` |
| `_api_json_response(...)` | `HttpResult api_json_response(const Json&, int status = 200)` |
| `_reject_unsupported_request_fields(...)` | `void reject_unsupported_request_fields(const Json&, const std::set<std::string>&, const std::string&)` |
| `_pick(data, names, default)` | `Json pick(const Json&, std::initializer_list<std::string_view>, const Json& = nullptr)` |
| `_api_float(...)` | `double api_float(const Json&, std::initializer_list<std::string_view>, std::optional<double>, const std::string&)` |
| `_api_int(...)` | `long long api_int(const Json&, std::initializer_list<std::string_view>, std::optional<long long>, const std::string&)` |
| `_api_strict_bool(...)` | `bool api_strict_bool(const Json&, std::initializer_list<std::string_view>, std::optional<bool>, const std::string&)` |
| `_reload_config_if_changed()` | `void reload_config_if_changed()` |
| `_write_json(path, data)` | `void write_json(const std::filesystem::path&, const Json&)` |
| `_query_dataset_list()` | `Json query_dataset_list()` |
| `_query_dataset_time_list(data)` | `Json query_dataset_time_list(const Json&)` |
| `_query_data_type_meta_list()` | `Json query_data_type_meta_list()` |
| `_run_portal_job(params)` | `JobResult run_portal_job(Json& params)` |
| `_portal_2d_params` / `_portal_3d_params` / ... | 同名 `portal_2d_params` / `portal_3d_params` ... |
| `_build_*_response` | 同名 `build_*_response` |

### 3.4 注册与分派

```cpp
struct PluginEntry {
    int sn;
    Handler handler;   // 接收原始 payload；内部调用 handle_portal_payload
};

const std::map<std::string, PluginEntry>& plugin_table();

HttpResult dispatch(const Json& payload);          // POST /
HttpResult handle_plugin(const std::string& name, const Json& payload); // 单插件路由
```

- `plugin_table()` 由 `registry.cpp` 静态初始化，内容必须与 `registry.py` 完全一致。
- `dispatch` 严格复刻 `resolve_portal_handler` 的校验顺序与错误文案。
- 每个单插件路由直接调用 `handler(payload)`，不校验 SN。

### 3.5 HTTP 层

```cpp
httplib::Server svr;
svr.new_task_queue = [] { return new httplib::ThreadPool(8); };

svr.Post("/", [](const httplib::Request& req, httplib::Response& res) {
    auto payload = parse_request_body(req.body, req.get_header_value("Content-Type"));
    auto result = jd::dispatch(payload);
    res.status = result.status;
    res.set_content(result.body.dump(), result.content_type.c_str());
});

for (const auto& [plugin, entry] : jd::plugin_table()) {
    auto route = "/" + plugin; // 注意：部分路由名与 Plugin 不同，见映射表
    svr.Post(route, ...);
}
// 特殊：/AnalyseReverberationWaveform、/waveguide/analyse
```

> 注意：路由名不一定等于 Plugin 名（例如 `AnalyseIce2DTLRD` 对应文件 `ice_2d_tl.py` 路由 `/AnalyseIce2DTLRD`；`AnalyseReverberation` 还有额外路由 `/AnalyseReverberationWaveform`）。以映射表 `route` 列为准。

### 3.6 请求体兼容

`_request_payload` 逻辑：
1. `Content-Type` 含 `application/json` → 解析 JSON，失败返回 `{}`。
2. 否则尝试表单；表单为空再尝试 JSON；失败返回 `{}`。
3. 非对象（数组/标量）→ `{}`。

C++ 里可先只实现 JSON（P0），表单兼容作为可选 TODO。

---

## 4. 关键差异与风险

1. **SN 冲突/不一致**
   - 统一入口以 `registry.py` 为准。
   - 部分文件内 `SN` 常量与 registry 不一致（仅用于 OpenAPI 示例，不参与校验）：
     - `ice_under_surface_reverberation.py`: 文件 `SN = 22`，registry `AnalyseIceReverberation = 16`
     - `ice_waveguide.py`: 文件 `SN = 25`，registry `AnalyseIceWaveguide = 16`
     - `waveguide.py`: 文件 `SN = 24`，registry `AnalyseWaveguide = 16`
   - C++ 只维护一份 registry，避免复制这些历史不一致。

2. **numpy 数值一致性**
   - 输出保留 `PORTAL_DECIMAL_PLACES = 6`；数组的 `round`、`nan → 9999.0` 等规则需逐接口对齐。
   - C++ 建议引入 Eigen（可选）处理矩阵/插值；纯标量逻辑用 STL。

3. **外部声学程序**
   - bellhop/kraken/ram 为 exe；C++ 需用 `CreateProcess`/`std::system` 调用，并复刻输入文件生成与输出解析。
   - 输入/输出文件、`jobs/<id>` 目录、状态文件 `job_status.json` 需保持一致。

4. **数据源依赖**
   - 环境数据：HYCOM NetCDF / NCSS、地形文件（`TopographyReader`）。
   - 数据目录：`catalog_queries` 走 SQL 数据库。
   - 天气/AIS：HTTP(S) 远程接口。
   - C++ 需要 netCDF-c/HDF5、MySQL/PostgreSQL 连接器、cpp-httplib SSL 客户端。

5. **配置与热加载**
   - `_reload_config_if_changed()` 有全局状态；C++ 建议 `ConfigManager` + `std::shared_mutex`，避免多线程竞态。

6. **作业目录副作用**
   - 带 `job_id` 的接口会写 `jobs/<job_id>/response_<plugin>.json`；C++ 需保持相同路径与命名。

7. **OpenAPI 文档**
   - 原接口通过 `openapi_extra` 生成文档；C++ 若不需要 Swagger，可先忽略，仅保留路由与示例常量。

8. **异步/线程模型**
   - Python 用 `asyncio.to_thread` + FastAPI；C++ 用 httplib ThreadPool 即可，注意计算任务不要阻塞 I/O 线程池过大。

---

## 5. 接口映射表（汇总）

完整可机读版本见 `endpoint_mapping.csv`。复杂度：
- `L1` 静态/简单查询
- `L2` 数据源/数据库/HTTP 简单读取
- `L3` numpy/数值服务
- `L4` 声学 exe + NetCDF
- `L5` 大规模编排/多模型/网络

| Plugin | SN | 路由 | 源文件 | 主要依赖 | 复杂度 | 建议批次 |
|---|---|---:|---|---|---|---|
| AnalyseRayTracing | 16 | /AnalyseRayTracing | analyse_ray_tracing.py | core 3D ray + bellhop3d | L4 | P4 |
| Analyse3DTLRD | 16 | /Analyse3DTLRD | analyse_3d_tl_rd.py | core 3D TL | L4 | P4 |
| Analyse2DTLRD | 16 | /Analyse2DTLRD | analyse_2d_tl_rd.py | core 2D TL | L4 | P4 |
| Analyse2DRTRD | 19 | /Analyse2DRTRD | analyse_2d_rt_rd.py | core 2D ray | L4 | P4 |
| Analyse2DRTRDDynamic | 19 | /Analyse2DRTRDDynamic | analyse_2d_rt_rd_dynamic.py | core + ray_animation | L4 | P4 |
| Analyse3DRTRDDynamic | 19 | /Analyse3DRTRDDynamic | analyse_3d_rt_rd_dynamic.py | core + ray_animation | L4 | P4 |
| AnalyseIce3DTLRD | 16 | /AnalyseIce3DTLRD | ice_3d_tl.py | core ice 3D TL | L4 | P4 |
| AnalyseIce2DTLRD | 16 | /AnalyseIce2DTLRD | ice_2d_tl.py | core ice 2D TL | L4 | P4 |
| AnalyseIce2DRTRD | 19 | /AnalyseIce2DRTRD | ice_2d_rt.py | core ice 2D ray | L4 | P4 |
| AnalyseIce2DRTRDDynamic | 19 | /AnalyseIce2DRTRDDynamic | ice_2d_rt_dynamic.py | core + ray_animation | L4 | P4 |
| AnalyseIce3DRTRDDynamic | 19 | /AnalyseIce3DRTRDDynamic | ice_3d_rt_rd_dynamic.py | core + ray_animation + thickness | L4 | P4 |
| GetEnvDataStation | 22 | /GetEnvDataStation | get_env_data_station.py | core env profiles/topo | L2 | P2 |
| GetEnvDataRoute | 110 | /GetEnvDataRoute | get_env_data_route.py | core env route | L2 | P2 |
| GetEnvDataRegion | 145 | /GetEnvDataRegion | get_env_data_region.py | core env region | L2 | P2 |
| GetEnvDataSourceList | 1 | /GetEnvDataSourceList | get_dataset_list.py | hycom_service.dataset_list | L1 | P1 |
| FetchTimeToList | 2 | /FetchTimeToList | get_dataset_time_list.py | hycom_service.time_list | L2 | P1 |
| GetDataTypeMetaInfo | 1 | /GetDataTypeMetaInfo | get_data_type_meta_list.py | catalog_queries (DB) | L2 | P1 |
| AnalyseEnvironmentNoise | 16 | /AnalyseEnvironmentNoise | analyse_env_noise.py | env noise services + distributed_noise | L5 | P5 |
| AnalyseKraken2DTLRD | 14 | /AnalyseKraken2DTLRD | kraken_2d_tl.py | core 2D TL + kraken | L4 | P4 |
| AnalyseKraken3DTLRD | 15 | /AnalyseKraken3DTLRD | kraken_3d_tl.py | core 3D TL + kraken | L4 | P4 |
| AnalyseIceKraken2DTLRD | 16 | /AnalyseIceKraken2DTLRD | ice_kraken_2d_tl.py | core ice 2D TL + kraken | L4 | P4 |
| AnalyseIceKraken3DTLRD | 17 | /AnalyseIceKraken3DTLRD | ice_kraken_3d_tl.py | core ice 3D TL + kraken | L4 | P4 |
| Analyse2DTLRDPE | 16 | /Analyse2DTLRDPE | ram_2d_tl.py | core 2D TL + ram | L4 | P4 |
| AnalyseRam3DTLRD | 19 | /AnalyseRam3DTLRD | ram_3d_tl.py | core 3D TL + ram | L4 | P4 |
| AnalyseIce2DTLRDPE | 16 | /AnalyseIce2DTLRDPE | ice_ram_2d_tl.py | core ice 2D TL + ram | L4 | P4 |
| AnalyseIceRam3DTLRD | 21 | /AnalyseIceRam3DTLRD | ice_ram_3d_tl.py | core ice 3D TL + ram | L4 | P4 |
| AnalyseReverberation | 16 | /AnalyseReverberation | open_water_reverberation.py | surface_reverberation_common | L3 | P5 |
| AnalyseReverberation | 16 | /AnalyseReverberationWaveform | open_water_reverberation_waveform.py | surface_reverberation_common | L3 | P5 |
| AnalyseIceReverberation | 16 | /AnalyseIceReverberation | ice_under_surface_reverberation.py | surface_reverberation_common | L3 | P5 |
| AnalyseBottomSurfaceReverberation | 23 | /AnalyseBottomSurfaceReverberation | bottom_surface_reverberation.py | surface_reverberation_common | L3 | P5 |
| AnalyseWaveguide | 16 | /AnalyseWaveguide | waveguide.py | services.waveguide + kraken | L3 | P3 |
| AnalyseIceWaveguide | 16 | /AnalyseIceWaveguide | ice_waveguide.py | services.waveguide + kraken | L3 | P3 |
| AnalyseWaveguideDecay | 16 | /AnalyseWaveguideDecay | analyse_waveguide_decay.py | services.waveguide(+decay) | L3 | P3 |
| AnalyseSoundLayer | 16 | /AnalyseSoundLayer | analyse_sound_layer.py | env + services.sound_layer | L3 | P3 |
| AnalyseSoundSpeedClass | 16 | /AnalyseSoundSpeedClass | analyse_sound_speed_class.py | env + services.sound_speed_class | L3 | P3 |
| AnalyseChannelAxis | 16 | /AnalyseChannelAxis | analyse_channel_axis.py | env + sound_speed_profile_emd_vmd | L3 | P3 |
| AnalyseSoundPropagationLoss | 16 | /AnalyseSoundPropagationLoss | analyse_sound_propagation_loss.py | core 3D TL + statistics | L4 | P4 |
| AnalyseChannelImpulseResponse | 24 | /AnalyseChannelImpulseResponse | analyse_channel_impulse_response.py | services.channel_impulse_response | L3 | P5 |
| AnalyseArcticMixedLayer | 16 | /AnalyseArcticMixedLayer | analyse_arctic_mixed_layer.py | env + services.arctic_mixed_layer | L3 | P3 |
| AnalyseReverbDecay | 16 | /AnalyseReverbDecay | analyse_reverb_decay.py | env + reverb_decay + reverb common | L3 | P5 |
| GetWeatherForecast | 146 | /GetWeatherForecast | get_weather_forecast.py | services.open_meteo_weather (HTTP) | L2 | P1(可选) |
| GetRealtimeAIS | 147 | /GetRealtimeAIS | get_realtime_ais.py | services.ais_realtime (HTTP) | L2 | P1(可选) |
| AnalyseForecastEnvironmentNoise | 148 | /AnalyseForecastEnvironmentNoise | analyse_forecast_environment_noise.py | forecast noise + weather | L5 | P5 |
| GetRegionalWeatherForecast | 149 | /GetRegionalWeatherForecast | get_regional_weather_forecast.py | open_meteo + regional_weather | L2 | P1(可选) |
| （无 Plugin） | - | /waveguide/analyse | waveguide_dispersion.py | waveguide_dispersion.analyse_waveguide | L3 | P3 |

### 辅助模块（不直接暴露 Plugin）

| 源文件 | 作用 | C++ 对应 |
|---|---|---|
| common.py | 协议适配、路由工厂 | `portal_protocol.*`, `http_server.*` |
| registry.py | Plugin/SN 表 | `registry.*` |
| dispatcher.py | 统一入口校验分派 | `dispatcher.*` |
| distributed_noise.py | 分布式噪声源构建 | `services/noise/*` |
| surface_reverberation_common.py | 海面/海底混响公共计算 | `services/reverberation/*` |

---

## 6. 建议的转换批次

### P0：协议层（先做）
- `registry`、`dispatcher`、`portal_protocol`、`http_server`、`main`。
- 所有 handler 先返回 `501 Not Implemented`（wrapped 时 `ResultCode=1`）。
- 验收：`POST /` 的 SN/Plugin 校验、wrapped/flat 包装、错误响应与 Python 一致。

### P1：查询类（试点业务）
- `GetEnvDataSourceList`（静态列表，最容易）。
- `FetchTimeToList`（HYCOM NCSS/时间列表，依赖 netCDF/HTTP）。
- `GetDataTypeMetaInfo`（数据库）。
- 可选：`GetWeatherForecast`、`GetRegionalWeatherForecast`、`GetRealtimeAIS`（纯 HTTP 客户端）。

### P2：环境数据类
- `GetEnvDataStation`、`GetEnvDataRoute`、`GetEnvDataRegion`。
- 需要 netCDF 读取、地形采样、网格插值、缺测值规则。

### P3：数值分析类
- `AnalyseSoundLayer`、`AnalyseSoundSpeedClass`、`AnalyseChannelAxis`、`AnalyseArcticMixedLayer`、`AnalyseWaveguideDecay`、`AnalyseWaveguide`、`AnalyseIceWaveguide`、`/waveguide/analyse`。
- 需要移植 numpy 算法（可选 Eigen）。

### P4：声传播类
- 全部 2D/3D TL/Ray（含 ice、kraken、ram、ray tracing、propagation loss）。
- 需要外部 exe 调用 + NetCDF 解析 + 作业目录/状态文件。

### P5：大规模编排类
- 环境噪声、混响、混响衰减、信道冲激响应、预报噪声。
- 依赖多个服务、并行任务、风/雨/航运噪声模型。

---

## 7. 映射 CSV 字段说明

`endpoint_mapping.csv` 列：

| 列 | 含义 |
|---|---|
| `file` | 源 Python 文件 |
| `plugin` | 门户 Plugin 名（空表示无 Plugin） |
| `sn` | 统一入口校验用 SN（以 registry 为准） |
| `route` | HTTP 路由 |
| `core_helpers` | 调用的 `backend_worker` 函数 |
| `services` | 调用的 `services/*` 或项目内模块 |
| `external` | 外部依赖（exe / NetCDF / DB / HTTP） |
| `complexity` | L1–L5 |
| `phase` | 建议批次 P0–P5 |
| `notes` | 备注（SN 不一致、额外路由等） |

---

## 8. 待确认

1. 本轮“部分接口”具体选哪一批？建议 **P0 + P1（3 个查询接口）** 先落地。
2. 是否需要保留 `openapi_extra` / Swagger 文档？若不需要，C++ 侧省略。
3. 环境数据/声学模型是否允许 C++ 通过 **调用现有 Python 服务或 exe** 复用，而不是完全重写算法？
4. 数据库/NetCDF/HTTP 的具体依赖库是否已确定（netCDF-c / HDF5 / MySQL Connector / OpenSSL）？
