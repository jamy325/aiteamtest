# AI 临摹式矢量重建系统设计方案

## 1. 核心定位

本系统的定位是 **Autonomous Vector Reconstruction Engine**，而不是“绘图程序 + AI 建议 + 用户点击应用”。

目标不是简单地根据轮廓点拟合曲线，而是让 AI 像熟练设计师一样，把输入位图当作参考层，结合语义判断、几何求解、拓扑约束和环境反馈，自主完成矢量重建。

核心分工如下：

- AI / Policy 层负责理解图形语义、提出修改意图、评估风险和决定是否继续自动执行。
- 传统算法负责精确求解几何参数，不让 AI 直接输出精确几何。
- 约束与拓扑系统负责维持闭合、连续、共切、重合、自交检查等一致性。
- 评分与预览系统负责验证修改是否带来正收益。
- Exporter 负责把最终 VectorDocument 输出为 SVG / DXF / JSON。

人工能力仍然保留，但定位已经降级为：

- optional debug UI
- external override
- external decision API
- 异常兜底或回放观察

它们不是引擎主流程依赖，也不是默认控制面。

## 2. 主流程

系统主流程应稳定为：

```text
Input Image
  -> Extraction
  -> Initial Draft
  -> AI / Policy Review
  -> Command Planning
  -> Preview / Validation
  -> DecisionPolicy
  -> Auto Apply / Auto Reject / Requires External Decision
  -> Export
```

对应展开如下：

1. 输入位图图像。
2. 自动提取 `binary_contours` 与 `skeleton_contours`。
3. 在 `ContourExtractor` 之后立即完成 `pixel -> vector` 坐标转换。
4. 自动生成初始矢量草图。
5. AI 使用结构化上下文审查当前结果，提出修改意图。
6. Planner 将意图整理为可验证的 command proposal。
7. Preview / Validation 层生成 overlay、diff、score、topology 和 constraint 检查结果。
8. `DecisionPolicy` 决定：
   - `auto_apply`
   - `auto_reject`
   - `requires_external_decision`
9. 对被接受的命令自动执行，对被拒绝的命令自动丢弃。
10. 对需要升级决策的命令交由外部决策方处理。
11. 最终导出 `SVG / DXF / JSON`。

这里的 `requires_external_decision` 不等于“必须人工点按钮”。它可以由以下任一消费者处理：

- 人类操作员
- 另一个 AI Agent
- CLI 工作流
- Web 服务
- 产品策略服务

## 3. 为什么不能只做轮廓拟合

传统轮廓拟合会出现这些典型问题：

1. 控制点过多。
2. 圆被拟合成波浪圆。
3. 圆弧出现鼓包。
4. 直线被拟合成轻微弯曲曲线。
5. 局部贴合但整体不自然。
6. 对噪声、锯齿和断裂敏感。
7. 无法理解同心、相切、对称、水平、垂直等几何语义。
8. 无法判断孔洞、实心区域和对象边界层级。
9. 只追求点云误差，不追求设计意义上的简洁表达。
10. 多路径之间的锚点吸附、重合和拼接关系容易丢失。
11. 修改后可能闭合，但会引入自交或拓扑污染。
12. 带透明度图像中，背景色可能污染颜色采样。

本质原因是：轮廓点只是像素采样结果，不等于图形语义。

因此正确方向不是“让用户在 UI 里不断点应用”，而是：

```text
位图
  -> 轮廓提取
  -> 坐标标准化
  -> 初始矢量
  -> AI 语义审查
  -> 命令规划
  -> 预览与验证
  -> 策略决策
  -> 自动执行或升级
  -> 导出
```

## 4. AI 审查输入与输出边界

AI 的职责是理解和建议，不是直接输出精确几何参数。

错误方式：

```json
{
  "tool": "replace_segment_with_arc",
  "cx": 123.4,
  "cy": 456.7,
  "r": 80.2
}
```

正确方式：

```json
{
  "tool": "propose_replace_segment_with_arc",
  "path_id": "path_001",
  "segment_range": [3, 6],
  "reason": "该区域视觉上应为标准圆弧，但当前被拟合成不稳定自由曲线。",
  "confidence": 0.91
}
```

AI 自主审核时应使用结构化上下文，而不是只看一张截图。典型输入包括：

- `VectorDocument JSON`
- overlay image
- diff image
- shape candidates
- proposed commands
- preview summary
- score breakdown
- topology findings
- constraint findings
- pipeline metadata

这保证 AI 能基于引擎状态做决策，而不是只基于表面像素做猜测。

## 5. 决策策略与升级

系统默认优先自动决策，而不是人工确认。

典型策略：

- 低风险、得分提升明确、拓扑安全的修改：`auto_apply`
- 收益不足、破坏约束或引入拓扑风险的修改：`auto_reject`
- 高风险、高不确定性或跨模块影响较大的修改：`requires_external_decision`

`requires_external_decision` 是一个策略升级点，不是 UI 按钮事件。

高风险操作应默认进入 policy-controlled escalation，例如：

- Bezier fallback
- 大范围批量替换
- 可能改变闭合关系或孔洞结构的命令
- 可能引入复杂度暴涨的自由曲线替换
- 低置信度但高影响范围的几何重写

其中 Bezier fallback 的定位尤其明确：

- 它是标准几何失败后的 fallback。
- 它不应抢占已高置信成功的 circle / ellipse / line / arc。
- 它默认不依赖 UI 人工确认。
- 它应进入策略控制的升级路径，由 external decision consumer 决定是否执行。

## 6. UI 的定位

UI 不是主流程控制器，而是可选 consumer。

UI 可以承担这些职责：

- debug 观察
- overlay / diff 可视化
- 单步回放
- 外部 override
- 异常诊断
- 演示与验收

UI 不应承担这些主流程依赖：

- “用户点击应用后命令才生效”
- “用户确认后 AI 才能继续下一步”
- “用户操作负责最终控制整个引擎”

引擎必须能在没有 UI 的情况下独立运行，并通过 CLI、服务调用或批处理工作流完成自动重建。

## 7. 总体架构

系统由以下模块组成：

1. `ImageProcessor`：图像预处理。
2. `ContourExtractor`：轮廓提取。
3. `CoordinateTransformer`：坐标系统转换。
4. `Resampler`：自适应重采样。
5. `InitialVectorizer`：初始矢量生成。
6. `VectorDocument`：纯数据型矢量文档结构。
7. `ObjectGraph`：对象层语义结构。
8. `ConstraintGraph`：几何约束系统。
9. `GlobalSnappingEngine`：跨路径锚点吸附与重合约束推断。
10. `BreakPointOptimizer`：切分点优化器。
11. `Renderer`：矢量渲染器。
12. `DistanceFieldDiffRenderer`：距离场差异图。
13. `SelfIntersectionDetector`：路径自交检测器。
14. `Scorer`：综合评分系统。
15. `AIAgent`：视觉 AI 审查与意图生成。
16. `BatchCommandPlanner`：AI 批量提议规划器。
17. `RefinementEngine`：鲁棒算法精化引擎。
18. `RefinementFeedback`：算法反哺 AI 的确定性反馈。
19. `SharedTangentOptimizer`：G1 共切同步优化器。
20. `SegmentRigidityPolicy`：混合路径刚性策略。
21. `TopologyEngine`：路径闭合、自交检测和拓扑一致性维护。
22. `AlphaAwareStyleAnalyzer`：颜色、透明度与样式分析。
23. `CommandExecutor`：命令执行器。
24. `BatchCommandExecutor`：批量命令执行器。
25. `HistoryManager`：撤销与回滚。
26. `Exporter`：`SVG / DXF / JSON` 导出。

这些模块的技术作用不变，但它们组合出来的产品形态应是自主引擎，而不是交互式绘图程序。

## 8. 引擎级执行视图

从执行视角看，完整链路如下：

```text
Input Image
  -> ImageProcessor
  -> ContourExtractor
  -> CoordinateTransformer
  -> Resampler
  -> InitialVectorizer
  -> VectorDocument / ObjectGraph / ConstraintGraph bootstrap
  -> AI Review with structured context
  -> BatchCommandPlanner
  -> Preview / Validation / Score / Topology checks
  -> DecisionPolicy
  -> CommandExecutor / BatchCommandExecutor
  -> RefinementFeedback
  -> Iteration or escalation
  -> Exporter
```

其中：

- `Preview / Validation` 负责证明“这次修改值得执行”。
- `DecisionPolicy` 负责决定“是继续自动化，还是升级给外部决策方”。
- `RefinementFeedback` 负责把确定性验证结果反哺给后续 AI / Policy 轮次。

## 9. 核心原则

### 9.1 所有核心算法运行在 Vector Space

一旦轮廓被提取，后续重建、评分、预览、约束、拓扑和导出都应统一在 Vector Space 中完成。

### 9.2 VectorDocument 是纯数据层

`VectorDocument` 负责表达对象、路径、锚点、几何段、样式和元数据，不直接承载 UI、OpenCV 或 AI SDK 依赖。

### 9.3 AI 只输出修改意图

AI 负责提出 intent，几何求解必须交由传统算法、约束系统和验证系统完成。

### 9.4 自动化优先，人工介入降级

默认路径应是自动审查、自动验证、自动决策、自动执行。人工只在策略升级、调试或 override 时出现。

### 9.5 评分、约束和拓扑是硬边界

任何提议即使“看起来像对的”，只要破坏评分、约束或拓扑边界，都不应直接进入自动执行。

## 10. 结论

这个系统不应继续被描述成“用户在 UI 中看 AI 建议并点击应用”的工具。

更准确的表述应是：

> 一个以位图为参考层、以 Vector Space 为统一计算空间、以 AI 语义审查和策略决策为驱动、以传统算法完成精确几何求解、以评分/约束/拓扑验证为闭环的 Autonomous Vector Reconstruction Engine。

人工能力、UI 和外部服务依然重要，但它们的角色是：

- optional consumer
- external decision maker
- debug and override surface

而不是核心重建流程的前提条件。
