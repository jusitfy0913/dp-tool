## ADDED Requirements

### Requirement: Frontend Pure Logic Extracted From Large Modules

标注页的纯计算逻辑 SHALL 抽成独立 ES Module，模块内 MUST NOT 访问 DOM、页面状态或网络。

#### Scenario: 模块无副作用

- **WHEN** 在 Node 环境中导入任一被抽出的纯逻辑模块
- **THEN** 导入 SHALL NOT 抛异常，且 MUST NOT 访问 `document`、`window`、`localStorage` 或发起网络请求

#### Scenario: 分档函数可单测

- **WHEN** 以任意置信度调用分档函数
- **THEN** 返回值 SHALL 只由入参与模块内常量决定，相同入参多次调用结果一致

#### Scenario: 历史栈深度受限

- **WHEN** 快照数量超过深度上限后继续压栈
- **THEN** 最旧的快照 SHALL 被丢弃，撤销栈长度 MUST NOT 超过上限

#### Scenario: 几何计算沿用统一坐标系

- **WHEN** 计算命中测试、吸附或包围盒
- **THEN** 输入输出 SHALL 使用图像坐标加显式变换参数，MUST NOT 依赖全局缩放状态

#### Scenario: 导入计划可预演

- **WHEN** 用扫描结果与类别映射生成导入计划
- **THEN** 该计划 SHALL 是一个纯数据对象，包含参与导入的图片数、分档计数、单例计数与阻塞原因；
  在用户确认之前 MUST NOT 修改页面状态或写盘

#### Scenario: 页面行为不变

- **WHEN** 拆分完成后打开标注页执行标注、撤销、导入、导出
- **THEN** 行为与拆分前一致，`index.html` MUST NOT 需要修改

#### Scenario: 零构建保持

- **WHEN** 拆分完成后在未安装 Node 的机器上打开网页
- **THEN** 页面 SHALL 正常加载，不得要求任何构建步骤

### Requirement: Full Chain Self-Check

仓库 SHALL 提供一条在进程内走完「标注 → 导出 → 数据集 → 产物 → 模型包 → 推理」的自检，MUST NOT 依赖 GPU、真实推理动态库或人工启动服务。

#### Scenario: 进程内执行

- **WHEN** 运行全链路自检脚本
- **THEN** 它 SHALL 直接以 ASGI 传输调用应用，MUST NOT 需要先启动独立服务或占用端口

#### Scenario: 每一步都验证

- **WHEN** 自检执行到某一环节
- **THEN** 它 SHALL 断言该环节的状态码与关键字段，并在失败时打印该环节的响应体

#### Scenario: 缺少 GPU 也能跑

- **WHEN** 运行环境没有可用的推理动态库或 GPU
- **THEN** 自检 SHALL 使用假引擎桩完成推理环节，MUST NOT 因此跳过该环节

#### Scenario: 可单步执行

- **WHEN** 调用者指定只运行某一个环节
- **THEN** 自检 SHALL 只执行该环节及其必要前置，并如实报告结果

#### Scenario: 进入 CI

- **WHEN** 后端测试集合被运行
- **THEN** 全链路自检 SHALL 被收录并与其他脚本一起给出通过/失败结论

### Requirement: Optional Browser End-To-End Script

仓库 SHALL 提供一个可选的浏览器端到端脚本，用于发版前人工验证，且 MUST NOT 成为主持续集成的必要步骤。

#### Scenario: 页面冒烟

- **WHEN** 运行该脚本
- **THEN** 它 SHALL 自行启动临时服务并加载标注页，校验前端脚本无报错、关键控件齐全、可在标注与推理页之间切换；
  完整业务链路由进程内全链路自检覆盖，该脚本 MUST NOT 要求真实模型或 GPU

#### Scenario: 截图留证

- **WHEN** 任一检查项失败
- **THEN** 脚本 SHALL 保存失败现场截图与页面控制台摘要到运行目录

#### Scenario: 不阻塞主 CI

- **WHEN** 主持续集成运行
- **THEN** 它 MUST NOT 调用该脚本，也不得要求安装浏览器
