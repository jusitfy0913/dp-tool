## MODIFIED Requirements

### Requirement: Anomaly Output Parsing
TRTForge SHALL parse EfficientAd, PaDiM, PatchCore, and AnomalyDINO TensorRT outputs into anomaly map and anomaly score by resolving outputs **by name first, then by element count**, and the C++ DLL and the Python-side helper MUST apply the same rule.

#### Scenario: 先按名字匹配
- **WHEN** an output tensor name contains `score`
- **THEN** the system SHALL treat it as the anomaly score
- **AND** when a name contains `anomaly_map`, or contains `map` but not `mask`, it SHALL be treated as the anomaly map

#### Scenario: 元素数最多者作为异常图
- **WHEN** any output has more elements than the currently selected anomaly map
- **THEN** that output SHALL replace the anomaly map selection
- **AND** an engine whose extra output is larger than the real map MUST be treated as a contract violation, not silently rendered

#### Scenario: 单元素输出补位成分数
- **WHEN** no output name contains `score`
- **THEN** an output with exactly one element whose name does not contain `label` SHALL be used as the score

#### Scenario: One output tensor
- **WHEN** the engine returns exactly one output tensor
- **THEN** the system SHALL treat it as the anomaly map and derive the score from the map maximum

#### Scenario: Two output tensors
- **WHEN** the engine returns two output tensors
- **THEN** the system SHALL resolve them by name first, then by element count
- **AND** when neither name matches, the larger (element-count) output SHALL be the anomaly map and the single-element output SHALL be the score
- **AND** the result MUST NOT depend on the order of the two outputs

#### Scenario: Three output tensors
- **WHEN** the engine returns three or more output tensors, such as Anomalib's `pred_label`
- **THEN** the system SHALL keep the name-based selections above
- **AND** it MUST NOT fall back to a fixed positional rule such as "the third output is the map"

#### Scenario: AnomalyDINO 两个输出
- **WHEN** the engine returns `anomaly_map` `[1,1,gh,gw]` and `pred_score` `[1,1]` in either order
- **THEN** the system SHALL select `anomaly_map` as the map and `pred_score` as the score
- **AND** the result MUST NOT depend on the order of the two outputs

#### Scenario: 分数与异常图是同一个张量
- **WHEN** the score selection resolves to the same tensor as the anomaly map
- **THEN** the system SHALL drop the explicit score and derive it from the map maximum
