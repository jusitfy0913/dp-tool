## ADDED Requirements

### Requirement: 模型包优先的模型选择

推理页 SHALL 在模型下拉中优先列出模型包，并把散装产物放在单独的兼容分组；选中模型包后 SHALL 自动填充引擎或 ONNX、标签、metadata，并 SHALL 在下拉项中显示校验结论。

#### Scenario: 下拉分组

- **WHEN** 模型产物目录中同时存在模型包与散装产物
- **THEN** 下拉列表 SHALL 按「模型包 / 训练产物（散装）」两组分别列出，模型包组排在前面

#### Scenario: 选中模型包自动填充

- **WHEN** 用户从模型包分组选中一项
- **THEN** 界面 SHALL 填入包内模型文件、标签文件与 metadata 的路径，并 SHALL 把这三项标为随包只读

#### Scenario: 显示校验结论

- **WHEN** 用户选中一个模型包
- **THEN** 界面 SHALL 显示该包的校验结论与原因；结论为 `block` 时 SHALL 禁用推理按钮并给出修复建议

#### Scenario: 结论为警告时仍可推理

- **WHEN** 模型包的校验结论为 `warn`
- **THEN** 界面 SHALL 显示警告内容，且推理按钮 SHALL 保持可用

#### Scenario: 切换到手工输入

- **WHEN** 用户在模型包分组选中一项后又手工修改了路径输入框
- **THEN** 界面 SHALL 视为散装用法，不再按模型包提交 `bundle_id`

#### Scenario: 没有模型包时不受影响

- **WHEN** 模型产物目录中没有任何模型包
- **THEN** 下拉列表 SHALL 只显示散装分组，界面行为与本变更之前一致
