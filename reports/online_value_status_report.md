# 在线特征空值状态追踪 — 实现报告

日期:2026-11-23
分支:HEAD(detached,基于 998e14bbc)

## 背景

线上取到的特征值为空时无法区分空在哪一步:实体不在库里、特征从未物化、
物化了但值为空、类型对不上、视图没注册,统统表现成一个 `None`。下游各自
补默认值且补法不一致。本次改动给 `get_online_features` 增加**可选的**
逐行逐特征状态判定,空值必须带原因;默认关闭,关闭时返回结构与现在完全一致。

## 改动清单

| 文件 | 改动 |
|---|---|
| `sdk/python/feast/online_status.py` | 新增。`ValueStatus` 枚举(互斥且穷尽,无"其他"兜底)、状态计算、默认填充、报告结构 |
| `sdk/python/feast/infra/online_stores/online_store.py` | 同步 `get_online_features` 接入状态路径:未注册视图预分流、分批+有上限重试、失败批次清单、注册表快照比对、报告装配 |
| `sdk/python/feast/repo_config.py` | 新增 `OnlineStatusConfig`(`online_status:` YAML 块),默认 `enabled: false` |
| `sdk/python/feast/online_response.py` | `OnlineResponse` 增加可选 `status_report` 旁挂属性(默认 `None`) |
| `sdk/python/feast/utils.py` | `_populate_response_from_feature_data` 增加 `emit_metrics` 开关;容忍读取失败哨兵行 |
| `sdk/python/feast/metrics.py` | 新增 `track_online_value_statuses`,复用现有 `feast_online_features_status_total{feature_view, status}` 计数器 |
| `sdk/python/tests/unit/online_store/test_online_value_status.py` | 新增 11 个测试 |
| `docs/reference/feature-store-yaml.md` | 新增 `online_status` 配置说明 |

## 设计要点(对应约束)

**状态分类互斥且穷尽。** 每个 (行, 特征) 恰好落在一个状态上,按优先级判定:
`feature_view_not_registered` → `store_unavailable` → `entity_not_found` →
`feature_not_materialized` → `null_value` → `unsupported_type` →
`outside_max_age` → `present`。没有 "other" 兜底:读取异常归入
`store_unavailable`(不是"缺失"),类型契约违例归入 `unsupported_type`。

**同一份快照。** 状态判定只使用本次请求已解析出的 `FeatureView` 对象和同一
次 `online_read` 返回的行,不二次查注册表、不增加在线存储往返。请求前后各取
一次注册表缓存快照 token(`cached_registry_proto.last_updated`),不一致时
`registry_version_mismatch=True` 并告警——报出版本差异,而不是静默用过期元
数据下结论。

**现有结构不变,状态旁挂。** `to_dict()` / `to_df()` / `to_arrow()` 的字段命名、
顺序、类型转换完全不变;状态挂在 `OnlineResponse.status_report` 上。能力关闭
时 `status_report is None`,返回与现在逐字节一致(有测试断言)。

**分批与重试。** `read_batch_size` 配置后才分批(默认不分,保持现有拆批策略);
每批重试上限 `max_batch_retries`(默认 2)。超限的批次进入
`report.failed_batches`(只含 feature_view、批次序号、行数、尝试次数、错误
类型——不含实体标识与特征值),该批行标记 `store_unavailable`,其余批次和
其余视图照常返回。

**默认值显式开启。** `fill_defaults: true` 时才填充;`OUTSIDE_MAX_AGE`(有真实
但过期的值)永不填充。填充后 `status_report` 仍保留原始状态,
`defaults_filled=True` 标记本次响应被填充过。

**日志卫生。** 新增日志只含 feature view 名、批次序号、行数、错误类型;
不出现特征值原文与实体标识原文。

**指标。** 缺失原因计数进入现有 `feast_online_features_status_total` 计数器,
`status` 标签取细粒度原因值,可按原因聚合;开启状态路径时旧的
present/not_found 粗粒度计数不再重复发射(避免双倍计数)。

## 范围说明

- 仅同步 `get_online_features`;`get_online_features_async` 与
  `precompute_online` 快速路径行为不变(不产出状态报告)。
- Redis/DynamoDB/Postgres 等覆盖了 `_read_features_per_fv` 的存储,在开启
  状态路径时走通用逐视图读取(功能正确,未使用其管道优化)。
- ODFV 转换特征不在状态矩阵内(矩阵覆盖基础特征视图)。

## 测试

### 新增测试(11 个,全部通过)

`sdk/python/tests/unit/online_store/test_online_value_status.py`,复用仓库现有
在线取值夹具(`CliRunner.local_repo` + `example_feature_repo_1.py`):

- `test_five_missing_scenarios` — 五种缺失形态逐条断言(要求五.1)
- `test_all_views_unregistered` — 全部视图未注册时仍返回行
- `test_partial_batch_failure` — 单批失败不拖垮整批,失败批次入清单(要求五.2)
- `test_store_unavailable_does_not_mark_missing` — 存储不可用报 unavailable 而非 missing(要求三.1)
- `test_failed_batch_retried_then_succeeds` — 上限内重试成功
- `test_duplicate_entities_read_once` — 重复实体只取一次、各位置一致(要求五.3)
- `test_metrics_aggregated_by_reason` — 指标按原因聚合计数(要求五.4)
- `test_fill_defaults_explicit_and_status_preserved` — 显式填充且原始状态保留
- `test_no_fill_by_default` — 默认不填充
- `test_registry_version_mismatch_reported` — 注册表快照变化被报出(要求三.2)
- `test_disabled_returns_legacy_response` — 关闭时返回结构与现在一致(要求一.10)

### 返回样例(要求五.2)

完整样例见 `reports/sample_output.json`(由 `reports/generate_sample.py` 生成)。

样例一:四种缺失形态 + 未注册视图,一次请求逐行逐特征判定:

```json
"features": {
  "lat": ["present", "null_value", "unsupported_type", "entity_not_found"],
  "lon": ["feature_not_materialized", "present", "feature_not_materialized", "entity_not_found"]
}
```

样例二:`read_batch_size=1` 时 driver 2 的批次持续失败,其余行照常返回,
失败批次明确列出(无实体标识):

```json
"values": {"driver_id": [1, 2, 3], "lat": [0.5, null, "oops"]}
"failed_batches": [{"feature_view": "driver_locations", "batch_index": 1,
                    "row_count": 1, "attempts": 2, "error_type": "RuntimeError"}]
"lat": ["present", "store_unavailable", "unsupported_type"]
```

### 延迟(要求四.3)

`reports/bench_status_overhead.py`,200 实体 × 2 特征,sqlite,本机:

```
status disabled (legacy): 1.95 ms/call (200 entities, 2 features)
status enabled           : 2.21 ms/call (200 entities, 2 features)
overhead: +0.26 ms/call (+13.2%)
```

开启后每次调用增加约 0.26ms(约 0.65µs/(行×特征) 的纯内存分类);关闭时
路径完全不变,零开销。

### 原有测试回归(要求五.5)

完整输出见 `reports/existing_online_retrieval_tests.log` 与
`reports/full_online_store_suite.log`。

```
tests/unit/online_store/test_online_retrieval.py: 12 passed, 4 skipped
tests/unit/online_store/ + tests/unit/test_metrics.py: 174 passed, 7 skipped
tests/unit/online_store/test_online_value_status.py: 11 passed
tests/unit/test_feature_server.py: 40 passed
```

(整套并发跑时 `test_five_missing_scenarios` 的 `feast apply` 子进程曾因机器
负载超时一次,单独重跑通过,属环境抖动;其余全绿。)

### 静态检查

```
ruff check / ruff format: 全部通过
mypy feast/online_status.py feast/online_response.py feast/metrics.py \
     feast/infra/online_stores/online_store.py feast/repo_config.py feast/utils.py:
Success: no issues found
```

## 环境备注

本机无 C 编译器,CI 锁定文件中的 `psutil==5.9.0` 与 `psycopg-c` 无法源码
构建,故测试环境用 `py3.12-minimal-requirements.txt`(去掉 `psycopg-c`)
加 `pytest*/sqlite_vec/mock/torch(cpu)` 搭建;这只影响测试环境搭建,
不影响代码改动本身。
