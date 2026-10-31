# web-studio-theme Specification

## Purpose
TBD - created by archiving change refactor-web-studio-theme. Update Purpose after archive.
## Requirements
### Requirement: Design Token System
前端样式 SHALL 通过 CSS 自定义属性集中定义语义色、尺度与效果三组设计令牌，页面中不得出现硬编码的颜色与间距字面量。

#### Scenario: 颜色与间距来自令牌
- **WHEN** 任一页面需要设置背景色、边框色、间距或圆角
- **THEN** 该样式 SHALL 引用 `:root` 中已定义的令牌变量，不得直接写颜色值或像素值

#### Scenario: 效果令牌可整体关闭
- **WHEN** 用户系统偏好为减少动效
- **THEN** 发光、扫描线与过渡效果 SHALL 通过效果令牌一次性归零，且语义色与尺度令牌不受影响

#### Scenario: 数字等宽对齐
- **WHEN** 页面展示分数、坐标、耗时等数值
- **THEN** 这些数值 SHALL 使用等宽字体与 tabular-nums，保证纵向对齐

### Requirement: Cyberpunk Industrial Visual Style
前端 SHALL 呈现工业赛博朋克视觉风格，同时 MUST 保证正文字号对比度不低于 WCAG AA 的 4.5:1。

#### Scenario: 对比度达标
- **WHEN** 正文文字以其所在面板的背景色为底渲染
- **THEN** 二者对比度 SHALL 不低于 4.5:1

#### Scenario: 装饰效果不覆盖图像区
- **WHEN** 页面绘制网格与扫描线等背景装饰
- **THEN** 这些装饰 SHALL 限制在页面底层，不得叠加在图像显示容器之上

#### Scenario: 日志区禁止闪烁
- **WHEN** 训练或引擎构建日志持续追加内容
- **THEN** 日志区域 SHALL NOT 使用闪烁、脉冲等持续性动效

### Requirement: Reusable Image Viewer Component
前端 SHALL 提供可复用的图像查看器组件，支持以光标为锚点的滚轮缩放与左键拖动平移。

#### Scenario: 滚轮以光标为锚点缩放
- **WHEN** 光标位于查看器内某图像像素上方并滚动滚轮
- **THEN** 该像素 SHALL 在缩放后仍停留在光标所在的屏幕位置，倍率被钳制在 0.1 至 20 之间

#### Scenario: 缩放不连带滚动页面
- **WHEN** 用户在查看器内滚动滚轮
- **THEN** 查看器 SHALL 阻止事件的默认行为，页面整体不得滚动

#### Scenario: 左键拖动平移
- **WHEN** 用户在查看器内按下左键并拖动
- **THEN** 图像 SHALL 随光标平移，且指针移出查看器边界后仍持续响应直至松开

#### Scenario: 视图复位
- **WHEN** 用户双击查看器，或点击工具条中的适应窗口按钮
- **THEN** 图像 SHALL 复位到完整可见的适应窗口状态，倍率回到基准值

#### Scenario: 倍率可读
- **WHEN** 查看器显示图像
- **THEN** 工具条 SHALL 显示当前倍率百分比，并提供 1:1 与放大、缩小操作

### Requirement: Toggleable Viewer Windows
查看器 SHALL 支持每个窗口独立开关显隐，并 SHALL 持久化用户的开关选择。

#### Scenario: 独立开关
- **WHEN** 用户切换某个窗口标题栏上的显隐开关
- **THEN** 仅该窗口显隐变化，同组其他窗口不受影响

#### Scenario: 开关状态跨会话保留
- **WHEN** 用户关闭并重新打开页面
- **THEN** 各窗口的显隐状态 SHALL 与上次选择一致

#### Scenario: 状态键不跨页面串扰
- **WHEN** 不同页面存在同名窗口
- **THEN** 其显隐状态 SHALL 通过带页面标识的键名分别存储，互不覆盖

#### Scenario: 全部关闭时的空状态
- **WHEN** 用户关闭了同组全部窗口
- **THEN** 页面 SHALL 显示空状态提示，不得留下空白网格

### Requirement: Side-by-Side Viewer Layout
多窗口 SHALL 在可视宽度允许时排布于同一行，且同一行内窗口等宽。

#### Scenario: 三窗口同排
- **WHEN** 视口宽度足以容纳三个窗口且最小窗口宽度为 300 像素
- **THEN** 三个窗口 SHALL 显示在同一行，且宽度相等

#### Scenario: 宽度不足时换行
- **WHEN** 视口宽度不足以容纳当前最小宽度下的全部窗口
- **THEN** 窗口 SHALL 自动换行，任一行内的窗口 SHALL 保持等宽

#### Scenario: 窗口过多时横向滚动
- **WHEN** 同组窗口数量超过一行的容纳上限
- **THEN** 容器 SHALL 提供横向滚动，不得压缩窗口高度或把窗口挤成窄条

