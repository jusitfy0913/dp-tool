## ADDED Requirements

### Requirement: HTTP Route Modules With Stable Contract

后端路由 SHALL 按功能域拆分为独立的 `APIRouter` 模块，且拆分 MUST NOT 改变任何对外 URL、请求体或响应体。

#### Scenario: 路由集合不变

- **WHEN** 重构完成后枚举全部已注册路由
- **THEN** `METHOD + PATH` 集合 SHALL 与重构前的快照逐条相等，缺失或新增都 SHALL 使检查失败

#### Scenario: 响应结构不变

- **WHEN** 前端以重构前的请求体调用任一接口
- **THEN** 响应字段与类型 SHALL 与重构前一致，前端代码 MUST NOT 需要修改

#### Scenario: 共享状态单点定义

- **WHEN** 多个路由模块需要访问任务管理器、流水线管理器或繁忙判定
- **THEN** 它们 SHALL 从同一个状态模块引用，MUST NOT 在各路由模块内重复创建实例

#### Scenario: 无循环导入

- **WHEN** 以任意顺序导入状态模块与路由模块
- **THEN** 导入 SHALL 成功，不得出现循环依赖

#### Scenario: 装配层只做装配

- **WHEN** 查看应用入口文件
- **THEN** 它 SHALL 只包含版本与常量、生命周期、异常处理、静态挂载与路由注册，
  且 MUST NOT 包含任何业务接口的实现

#### Scenario: 安装形态同样可用

- **WHEN** 以打包后的可执行程序启动
- **THEN** 全部路由 SHALL 正常注册与响应，不得因缺少模块而启动失败

### Requirement: Route Table Snapshot Guard

仓库 SHALL 保存一份路由表快照，并 SHALL 提供测试在其中逐条比对运行时路由。

#### Scenario: 快照检查通过

- **WHEN** 运行路由表测试且实现与快照一致
- **THEN** 测试 SHALL 通过并报告路由总数

#### Scenario: 路由被误删

- **WHEN** 实现中缺少快照里登记的一条路由
- **THEN** 测试 SHALL 失败并列出该条 `METHOD PATH`

#### Scenario: 路由被误加

- **WHEN** 实现中出现了快照里没有的路由
- **THEN** 测试 SHALL 失败并列出该条 `METHOD PATH`

#### Scenario: 显式重建快照

- **WHEN** 开发者确实需要新增或删除路由并传入重建参数
- **THEN** 测试 SHALL 重写快照文件，并在输出中列出所有变化项

#### Scenario: 文档类路由不参与比对

- **WHEN** 收集运行时路由
- **THEN** 交互式文档与 OpenAPI 描述路由 SHALL 被排除，静态资源挂载 SHALL NOT 被当作接口路由
