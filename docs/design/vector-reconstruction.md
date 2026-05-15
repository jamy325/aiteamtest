# AI 临摹式矢量重建系统 设计方案 

## 1. 核心定位

本系统目标不是简单地“根据轮廓点拟合曲线”，而是实现一种接近人工在 Photoshop / Illustrator / CAD 中临摹图形的矢量重建流程。

核心思想：

AI 负责判断图形语义，传统算法负责精确求解参数，约束系统负责保持几何关系，拓扑系统负责路径连续、闭合和自交检测，坐标系统负责统一计算和导出，用户操作负责最终控制。

系统最终应实现：

1. 输入位图图像。
2. 自动提取 binary_contours 和 skeleton_contours。
3. 在 ContourExtractor 后立即将 pixel 坐标转换为 Vector Space。
4. 自动生成初始矢量草图。
5. AI 审查当前矢量结果。
6. AI 只提出“修改意图”，不直接决定精确几何参数。
7. BreakPointOptimizer 优化切分点。
8. RefinementEngine 使用 RANSAC、最小二乘、约束优化等方法精确求解。
9. RefinementFeedback 将算法验证结果反哺 AI。
10. ConstraintGraph 保持水平、垂直、同心、相切、重合、连续性等关系。
11. GlobalSnappingEngine 推断跨 Path / 跨 Object 的锚点重合关系。
12. SharedTangentConstraint 维护相邻段 G1 共切关系。
13. TopologyEngine 维护路径闭合、混合路径连接和自交检测。
14. SegmentRigidityPolicy 决定拓扑修正时优先移动哪些几何段。
15. AlphaAwareStyleAnalyzer 处理颜色、透明度和背景污染。
16. Scorer 使用距离场、复杂度惩罚、约束违规、拓扑错误等指标评分。
17. 用户可锁定、辅助分段、拖点、回滚。
18. 最终导出 SVG / DXF / JSON。

---

## 2. 为什么不能只做轮廓拟合

传统轮廓拟合会出现：

1. 控制点过多。
2. 圆被拟合成波浪圆。
3. 圆弧出现鼓包。
4. 直线被拟合成轻微弯曲曲线。
5. 局部贴合但整体不自然。
6. 对噪声、锯齿、断裂非常敏感。
7. 无法理解同心、相切、对称、水平、垂直等几何语义。
8. 无法判断哪些轮廓是孔洞，哪些是实心区域。
9. 只追求点云误差，不追求人工设计意义上的简洁表达。
10. 坐标系统、单位、导出精度容易混乱。
11. 多条路径之间的端点吸附、重合、拼接关系容易丢失。
12. AI 或算法修改后，路径可能闭合但发生自交。
13. G1 连续如果只靠单边移动，容易破坏另一段的拟合质量。
14. 带透明度图像中，背景色可能污染颜色采样。

本质问题：

轮廓点只是像素采样结果，不等于图形语义。

正确方向：

位图 → 轮廓点 → 坐标标准化 → 初始矢量 → AI 语义审查 → 切分点优化 → 鲁棒算法精化 → 共享切线约束 → 约束校正 → 全局锚点吸附 → 拓扑闭合与自交检测 → 评分验证 → 用户确认 → 导出。

---

## 3. 总体架构

系统由以下模块组成：

1. ImageProcessor：图像预处理
2. ContourExtractor：轮廓提取
3. CoordinateTransformer：坐标系统转换
4. Resampler：自适应重采样
5. InitialVectorizer：初始矢量生成
6. VectorDocument：纯数据型矢量文档结构
7. ObjectGraph：对象层语义结构
8. ConstraintGraph：几何约束系统
9. GlobalSnappingEngine：跨路径锚点吸附与重合约束推断
10. BreakPointOptimizer：切分点优化器
11. Renderer：矢量渲染器
12. DistanceFieldDiffRenderer：距离场差异图
13. SelfIntersectionDetector：路径自交检测器
14. Scorer：综合评分系统
15. AIAgent：视觉 AI 审查与意图生成
16. BatchCommandPlanner：AI 批量提议规划器
17. RefinementEngine：鲁棒算法精化引擎
18. RefinementFeedback：算法反哺 AI 的确定性反馈
19. SharedTangentOptimizer：G1 共切同步优化器
20. SegmentRigidityPolicy：混合路径刚性策略
21. TopologyEngine：路径闭合、自交检测和拓扑一致性维护
22. AlphaAwareStyleAnalyzer：颜色、透明度与样式分析
23. CommandExecutor：命令执行器
24. BatchCommandExecutor：批量命令执行器
25. HistoryManager：撤销与回滚
26. Exporter：SVG / DXF / JSON 导出

整体流程：

```text
输入图片
  ↓
图像预处理
  ↓
提取 binary_contours 和 skeleton_contours
  ↓
pixel_to_vector 坐标转换
  ↓
后续核心算法全部在 Vector Space 中计算
  ↓
自适应重采样
  ↓
初始矢量拟合
  ↓
建立 Object / Path / Segment / Anchor 结构
  ↓
推断基础约束
  ↓
GlobalSnappingEngine 推断 coincident 约束
  ↓
检测 G1 continuity candidate
  ↓
渲染 Overlay 图
  ↓
生成 Distance Field Diff 图
  ↓
AI 审查并提出单个或批量修改意图
  ↓
BreakPointOptimizer 优化切分范围
  ↓
RANSAC 鲁棒精化
  ↓
最小二乘精拟合
  ↓
SharedTangentOptimizer 同步优化共切段
  ↓
RefinementFeedback 判断提议可信度
  ↓
SegmentRigidityPolicy 决定拓扑修正移动谁
  ↓
Path Closing 拓扑修正
  ↓
SelfIntersectionDetector 检测自交
  ↓
ConstraintGraph 约束校正
  ↓
Scorer 评分验证
  ↓
用户确认 / 自动迭代
  ↓
导出 SVG / DXF / JSON
```

---

## 4. 核心原则

### 4.1 AI 只判断意图，不直接给精确参数

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
  "reason": "该区域视觉上是标准圆弧，但当前被拟合成波浪样条",
  "confidence": 0.91
}
```

然后系统负责：

1. 获取该范围内的原始点。
2. 优化切分点。
3. 使用 RANSAC 拟合候选圆。
4. 剔除离群点。
5. 用内点重新最小二乘精拟合。
6. 计算圆弧起止角。
7. 应用相切、连续性、闭合约束。
8. 检查路径自交。
9. 评分验证修改是否有效。

原则：

AI 判断“是什么”，算法计算“是多少”。

---

### 4.2 所有核心算法只在 Vector Space 中运行

坐标流向必须单向、清晰：

```text
ContourExtractor 获取 pixel 坐标
→ CoordinateTransformer.pixel_to_vector
→ 后续拟合、评分、约束、拓扑全部在 Vector Space 中运行
→ 显示时 vector_to_pixel
→ SVG 导出时 vector_to_svg
→ DXF 导出时 vector_to_dxf
```

禁止在拟合、评分、约束求解中混用 pixel 坐标、SVG 坐标和 DXF 坐标。

---

### 4.3 VectorDocument 必须是纯数据层

VectorDocument 只能负责存储：

1. Document
2. Object
3. Path
4. Segment
5. Anchor
6. Constraint
7. Style
8. CoordinateSystem

VectorDocument 不应包含：

1. 渲染逻辑
2. 拟合算法
3. AI 调用
4. OpenCV 图像处理
5. PyQt 控件逻辑
6. 导出副作用

推荐工程结构：

```text
core/
  types.py
  document.py
  geometry.py
  constraints.py
  coordinate.py

services/
  image_processor.py
  contour_extractor.py
  vectorizer.py
  breakpoint_optimizer.py
  refiner.py
  shared_tangent.py
  topology.py
  snapping.py
  scorer.py
  renderer.py
  exporter.py
  ai_agent.py
  command_executor.py

ui/
  main_window.py
  canvas_widget.py

tests/
  test_coordinate.py
  test_document_json.py
  test_minimal_pipeline.py
```

---

### 4.4 鲁棒拟合优先于普通最小二乘

普通最小二乘容易被离群点带偏。

RefinementEngine 必须支持鲁棒回归：

1. RANSAC Line
2. RANSAC Circle
3. RANSAC Arc
4. RANSAC Ellipse
5. 离群点剔除
6. 内点重拟合
7. 置信度评估

---

### 4.5 约束优先于单段最优

单段拟合误差最小，不代表整体图形正确。

系统需要维护：

1. 水平
2. 垂直
3. 平行
4. 垂直
5. 共线
6. 同心
7. 等半径
8. 相切
9. 对称
10. 重合 coincident
11. G0 / G1 / G2 连续
12. 共享切线 shared tangent
13. 闭合路径首尾连接
14. 孔洞与外轮廓关系

---

### 4.6 拓扑一致性必须强制维护

任何编辑、替换、AI 修改后，都必须执行：

```text
enforce_path_topology(path)
detect_self_intersection(path)
```

确保：

1. segment[i].end == segment[i+1].start
2. 闭合路径 last_segment.end == first_segment.start
3. 小 gap 自动吸附
4. 大 gap 标记 topology_error
5. 自交路径标记为 self_intersected
6. SVG / DXF 输出不会出现缝隙和异常填充

---

### 4.7 用户操作优先于 AI

用户手动锁定的内容，AI 不能修改。

支持：

1. 锁定路径
2. 锁定分段
3. 锁定几何类型
4. 锁定控制点
5. 锁定约束
6. 撤销 AI 修改
7. 用户确认后再应用 AI 建议

---

## 5. Phase 1 编码顺序

进入第一行代码前，先实现最小核心结构，不要先写 UI 和复杂拟合。

推荐顺序：

```text
1. core/types.py
2. core/coordinate.py
3. core/document.py
4. services/contour_extractor.py
5. services/simple_vectorizer.py
6. services/json_exporter.py
7. tests/test_coordinate.py
8. tests/test_document_json.py
9. tests/test_minimal_pipeline.py
```

### 5.1 types.py

只定义纯数据结构，建议使用 dataclass。

核心类型：

1. Anchor
2. Segment
3. Path
4. Object
5. Constraint
6. Style
7. CoordinateSystem
8. VectorDocument

禁止依赖：

1. PyQt
2. Matplotlib
3. OpenCV
4. AI SDK
5. 文件系统副作用

---

### 5.2 coordinate.py

只负责坐标转换：

1. pixel_to_vector
2. vector_to_pixel
3. vector_to_svg
4. vector_to_dxf
5. px_to_mm
6. mm_to_px
7. y_axis_flip
8. precision_rounding

必须写单元测试。

---

### 5.3 document.py

只负责：

1. VectorDocument 创建
2. add_object
3. add_path
4. add_segment
5. add_anchor
6. add_constraint
7. to_json()
8. from_json()

禁止包含：

1. 拟合算法
2. 渲染逻辑
3. UI 状态
4. OpenCV 处理

---

### 5.4 minimal_pipeline.py

第一条可运行流水线只做：

```text
读取图片
→ OpenCV 提取轮廓
→ pixel_to_vector
→ 简单 line / bezier 分段
→ 存入 VectorDocument
→ 导出 JSON
```

目标不是拟合效果，而是先验证数据结构和坐标系统正确。

---

## 6. CoordinateTransformer 坐标系统

### 6.1 为什么需要

不同空间的坐标规则不同：

图像空间：

```text
原点：左上角
Y 轴：向下
单位：px
坐标：整数为主
```

SVG 空间：

```text
原点：左上角
Y 轴：向下
单位：用户单位
支持 viewBox
```

DXF / CAD 空间：

```text
原点：工程坐标
Y 轴：通常向上
单位：mm / inch
坐标：浮点
```

如果不统一管理，会出现：

1. 上下颠倒。
2. 尺寸不对。
3. 圆心偏移。
4. 半径单位错误。
5. 导出 DXF 后比例异常。
6. SVG 和 Matplotlib 显示不一致。
7. 用户拖点坐标和导出结果不一致。

---

### 6.2 VectorDocument 坐标定义

```json
{
  "coordinate_system": {
    "internal_space": "vector",
    "source_space": "pixel",
    "origin": "top_left",
    "y_axis": "down",
    "unit": "px",
    "precision": 4,
    "viewBox": [0, 0, 800, 600],
    "scale": {
      "px_to_mm": 0.1667
    }
  }
}
```

---

### 6.3 内部坐标原则

1. 内部统一使用 float。
2. 不在拟合阶段做整数化。
3. 只在显示或导出时做精度格式化。
4. DXF 导出时统一做单位换算。
5. SVG 导出时使用 viewBox 保持尺寸一致。
6. 所有算法输入都必须是 Vector Space 坐标。

---

## 7. VectorDocument 数据结构

### 7.1 层级结构

```text
Document
  └── Object
        └── Path
              └── Segment
                    └── Anchor / Handle
```

---

### 7.2 Document

```json
{
  "document_id": "doc_001",
  "width": 800,
  "height": 600,
  "coordinate_system": {},
  "objects": [],
  "paths": [],
  "segments": [],
  "anchors": [],
  "constraints": [],
  "metadata": {}
}
```

---

### 7.3 Object

```json
{
  "object_id": "obj_001",
  "type": "mechanical_part",
  "semantic_label": "outer_shape_with_hole",
  "paths": ["path_001", "path_002"],
  "constraints": ["constraint_001", "constraint_002"],
  "confidence": 0.86,
  "locked": false
}
```

---

### 7.4 Path

```json
{
  "path_id": "path_001",
  "object_id": "obj_001",
  "closed": true,
  "source": "binary_contour",
  "fill_role": "outer",
  "parent_path": null,
  "child_paths": ["path_002"],
  "segments": ["seg_001", "seg_002"],
  "style": {
    "fill_color": [32, 32, 32],
    "fill_alpha": 1.0,
    "stroke_color": null,
    "stroke_alpha": null,
    "stroke_width": 0,
    "opacity": 1.0,
    "color_confidence": 0.94,
    "color_variance": 3.2,
    "alpha_variance": 0.01,
    "paint_type": "solid"
  },
  "topology_status": "closed",
  "max_gap": 0.0,
  "self_intersection_count": 0,
  "locked": false
}
```

fill_role 可选：

```text
outer
hole
island
stroke
unknown
```

---

### 7.5 Segment

支持类型：

1. line
2. arc
3. circle
4. ellipse
5. bezier
6. bspline
7. polyline

内部角度单位约定：

```text
arc.start_angle   -> radians
arc.end_angle     -> radians
ellipse.rotation  -> radians
```

如果外部输入是 degree，必须通过显式导入适配或 `angle_unit='degree'`
转换，核心算法默认不会猜测角度单位。

```json
{
  "segment_id": "seg_001",
  "path_id": "path_001",
  "type": "arc",
  "params": {
    "cx": 300.0,
    "cy": 220.0,
    "r": 55.0,
    "start_angle": 0.3491,
    "end_angle": 2.618,
    "direction": "ccw"
  },
  "anchors": ["anchor_001", "anchor_002"],
  "fit_error": 1.2,
  "complexity_score": 0.3,
  "confidence": 0.91,
  "rigidity": "high",
  "locked": false
}
```

---

### 7.6 Anchor

```json
{
  "anchor_id": "anchor_001",
  "path_id": "path_001",
  "position": [120.0, 300.0],
  "continuity": "smooth",
  "shared_tangent": [0.7071, 0.7071],
  "locked": false,
  "in_handle": [110.0, 290.0],
  "out_handle": [130.0, 310.0]
}
```

continuity 可选：

```text
corner      G0，尖角
smooth      G1，切向连续
symmetric   G1 + 对称手柄
curvature   G2，曲率连续
```

MVP 阶段建议只做：

1. corner
2. smooth
3. symmetric

G2 后置。

---

## 8. ImageProcessor 与 ContourExtractor

### 8.1 输出两套轮廓

必须保留：

```text
binary_contours：用于区域边界、孔洞、填充语义、颜色采样
skeleton_contours：用于细线、中心线、线稿拟合
```

不能只保留 skeleton，否则会丢失：

1. 孔洞关系
2. 填充区域
3. 外轮廓与内轮廓嵌套
4. 颜色采样区域
5. SVG fill-rule 语义

---

### 8.2 轮廓提取流程

```text
灰度化
→ 二值化
→ 去噪
→ 形态学闭运算
→ 提取 binary contours
→ 提取 hierarchy
→ 可选骨架化
→ 提取 skeleton contours
→ 去重点
→ pixel_to_vector
→ 输出 Vector Space 轮廓点
```

---

### 8.3 轮廓数据结构

```json
{
  "contour_id": "contour_001",
  "source": "binary_contour",
  "points": [[x1, y1], [x2, y2]],
  "coordinate_space": "vector",
  "closed": true,
  "area": 12345,
  "depth": 0,
  "parent_contour": null,
  "children": ["contour_002"]
}
```

---

## 9. Resampler 自适应重采样

工业大图可能有数万个轮廓点，不能全部进入拟合。

### 9.1 目标

1. 直线区域减少采样。
2. 圆弧区域保留适中采样。
3. 尖角、高曲率区域增加采样。
4. 高频噪声过滤。
5. 防止初始拟合阶段卡死。
6. 保留几何关键点。

### 9.2 推荐流程

```text
Vector Space 原始轮廓点
→ 按弧长排序
→ 均匀重采样
→ 局部角度估计
→ 曲率估计
→ 直线区域抽稀
→ 高曲率区域保留
→ 噪声点过滤
→ 输出优化点集
```

---

## 10. Segment DOF 与刚性策略

### 10.1 Segment Rigidity

```text
line: high
circle: high
arc: high
ellipse: medium_high
bezier: medium
bspline: low
polyline: low
```

### 10.2 拓扑修正优先级

当 segment A 和 segment B 出现 gap：

1. 如果一方 locked，移动未锁定方。
2. 如果一方 rigidity 低，移动低刚性方。
3. 如果都是低刚性，取中点吸附。
4. 如果都是高刚性，只做最小量调整。
5. 如果都是 locked，标记 topology_error。
6. 如果存在约束，优先保持约束。

---

## 11. GlobalSnappingEngine 全局锚点吸附

### 11.1 目标

扫描所有 Anchor，发现距离小于 epsilon 的端点，并推断为 coincident 约束。

---

### 11.2 Coincident Constraint

```json
{
  "constraint_id": "constraint_087",
  "type": "coincident",
  "targets": ["anchor_001", "anchor_087"],
  "strength": "soft",
  "source": "algorithm_inferred",
  "confidence": 0.9,
  "locked": false
}
```

推荐 epsilon：

```text
epsilon = 1px ~ 3px 转换后的 Vector Space 距离
```

策略：

1. 用户锁定点不自动移动。
2. soft coincident 先记录，不强制吸附。
3. hard coincident 可直接吸附。
4. 高 confidence 可自动应用。
5. 中低 confidence 交给 AI 或用户确认。
6. 跨 Object 吸附需要更谨慎，避免误合并。

---

## 12. ConstraintGraph 约束系统

### 12.1 MVP 支持约束

1. horizontal：水平
2. vertical：垂直
3. parallel：平行
4. perpendicular：垂直
5. collinear：共线
6. concentric：共圆心
7. equal_radius：等半径
8. tangent：相切
9. symmetric：对称
10. coincident：重合
11. shared_tangent：共享切线
12. g1_continuity：G1 连续

---

### 12.2 Constraint 数据结构

```json
{
  "constraint_id": "constraint_001",
  "type": "shared_tangent",
  "targets": ["seg_001", "seg_002", "anchor_008"],
  "strength": "soft",
  "source": "algorithm_inferred",
  "confidence": 0.84,
  "locked": false
}
```

---

## 13. SharedTangentConstraint 共切同步优化

### 13.1 问题

当两个相邻段，例如 Line + Arc，被标记为 G1 连续时，不能简单地：

```text
先拟合 Arc
再移动 Line
```

这可能破坏 Line 的拟合，也可能让 Arc 偏离原始点。

---

### 13.2 正确策略

在 G1 连续锚点处，引入共享切线变量：

```json
{
  "anchor_id": "anchor_008",
  "continuity": "smooth",
  "shared_tangent": [tx, ty]
}
```

Line 和 Arc 同时受该切线约束：

```text
Line direction ≈ shared_tangent
Arc tangent at endpoint ≈ shared_tangent
```

---

### 13.3 局部双段优化目标

```text
loss =
    line_fit_error
  + arc_fit_error
  + λ * tangent_mismatch
  + μ * movement_penalty
```

MVP 只需要支持局部双段优化，不需要一开始实现全局 Solver。

---

## 14. BreakPointOptimizer 切分点优化器

### 14.1 输入

```json
{
  "path_id": "path_001",
  "rough_range": [120, 210],
  "target_type": "arc"
}
```

### 14.2 优化依据

1. 曲率突变点
2. 角度突变点
3. 拟合残差峰值
4. 切线方向突变
5. Distance Field Diff 热点
6. 用户辅助分段点
7. 相邻段端点
8. AI 标注区域

### 14.3 输出

```json
{
  "optimized_range": [128, 204],
  "breakpoints": [128, 204],
  "confidence": 0.87,
  "reason": "边界处存在曲率突变和拟合残差峰值"
}
```

---

## 15. RefinementEngine 鲁棒精化引擎

### 15.1 支持精化器

1. RANSACLineFitter
2. RANSACCircleFitter
3. RANSACArcFitter
4. RANSACEllipseFitter
5. PreciseLineFitter
6. PreciseCircleFitter
7. PreciseArcFitter
8. PreciseEllipseFitter
9. BezierOptimizer
10. ConstraintAwareFitter
11. SharedTangentOptimizer

---

### 15.2 RANSAC 圆弧精化流程

```text
AI 指定点集
→ BreakPointOptimizer 优化切分范围
→ 随机采样 3 点
→ 拟合候选圆
→ 计算所有点到圆的径向距离
→ 统计内点
→ 选择内点最多且误差最低的候选圆
→ 剔除离群点
→ 用内点重新最小二乘拟合圆
→ 计算圆弧起止角
→ 判断方向
→ 验证弧长角度范围
→ 应用相切 / 连续性 / 闭合约束
→ 输出 arc segment
```

### 15.3 RANSAC 验证条件

```text
min_arc_angle = 15° ~ 20°
min_inlier_ratio = 0.75
max_radial_error = 1.5 px
max_center_shift = 可配置
```

---

## 16. RefinementFeedback 确定性反馈环

### 16.1 执行闭环

```text
AI Propose：这里像圆弧
Algorithm Try：RANSAC 拟合
Validation：内点率只有 40%
Feedback：该提议可信度低，建议重新评估为 Bezier / 噪声 / 分段错误
```

### 16.2 Feedback 数据结构

```json
{
  "proposal_id": "proposal_001",
  "success": false,
  "reason": "low_inlier_ratio",
  "inlier_ratio": 0.40,
  "fit_error": 5.8,
  "suggestion": "try_bezier_or_adjust_breakpoints",
  "retry_policy": "ask_ai_reconsider"
}
```

---

## 17. TopologyEngine 路径闭合、自交与拓扑一致

### 17.1 enforce_path_topology

每次执行命令后调用：

```text
enforce_path_topology(path)
```

处理逻辑：

1. 检查相邻段端点距离。
2. 根据 SegmentRigidityPolicy 判断移动谁。
3. 小 gap 自动吸附。
4. 大 gap 标记 topology_error。
5. 闭合路径强制首尾连接。
6. 必要时微调锚点。
7. 记录修改影响范围。

---

### 17.2 SelfIntersectionDetector

路径即使没有 gap，也可能自交。

自交会影响：

1. SVG fill-rule
2. DXF 导出
3. 布尔运算
4. 颜色填充
5. 后续约束求解

检测方式：

1. 对 Line 直接检测线段交叉。
2. 对 Arc / Bezier / Spline 先采样成 polyline。
3. 使用 Shapely LineString.is_simple 或自定义线段交叉检测。
4. 闭合路径需忽略相邻线段共享端点导致的伪交叉。

---

### 17.3 自交拓扑状态

```json
{
  "path_id": "path_001",
  "topology_status": "self_intersected",
  "max_gap": 0.0,
  "self_intersection_count": 2,
  "self_intersection_points": [[120.4, 88.1], [132.7, 95.2]]
}
```

状态可选：

```text
closed
open
gap_detected
self_intersected
invalid
```

---

## 18. 填充语义与布尔关系

### 18.1 数据来源

必须使用 binary_contours 的 hierarchy，而不是 skeleton contours。

### 18.2 Path 填充结构

```json
{
  "path_id": "path_002",
  "closed": true,
  "fill_role": "hole",
  "parent_path": "path_001"
}
```

### 18.3 SVG 输出策略

推荐使用：

```xml
<path fill-rule="evenodd" d="..." />
```

或者导出为 compound path。

---

## 19. AlphaAwareStyleAnalyzer 颜色、透明度与样式分析

### 19.1 需要支持

1. fill_color
2. fill_alpha
3. stroke_color
4. stroke_alpha
5. stroke_width
6. opacity
7. color_variance
8. alpha_variance
9. paint_type
10. gradient_candidate
11. transparency_candidate

---

### 19.2 颜色采样问题

对于 PNG 或带透明背景的图像，背景色可能污染采样。

必须考虑：

1. Alpha 通道
2. 半透明边缘
3. 抗锯齿边界
4. 白底图像背景
5. 阴影和渐变

---

### 19.3 采样流程

```text
1. 根据 Path 生成内部 mask
2. 如果图像有 Alpha 通道，只采样 alpha > threshold 的像素
3. 对 mask 做 erosion，避开边缘抗锯齿
4. 采样内部像素
5. 使用 median color 或 dominant color
6. 计算 color_variance
7. 计算 alpha_variance
8. 方差低：认为是纯色
9. color_variance 高：标记为 gradient_candidate / unknown
10. alpha_variance 高：标记为 transparency_candidate
```

---

### 19.4 Style 数据结构

```json
{
  "style": {
    "fill_color": [32, 32, 32],
    "fill_alpha": 1.0,
    "stroke_color": null,
    "stroke_alpha": null,
    "stroke_width": 0,
    "opacity": 1.0,
    "color_confidence": 0.94,
    "color_variance": 3.2,
    "alpha_variance": 0.01,
    "paint_type": "solid"
  }
}
```

paint_type 可选：

```text
solid
gradient_candidate
linear_gradient
radial_gradient
texture
transparency_candidate
unknown
```

---

## 20. Distance Field Diff 距离场差异图

### 20.1 双向距离

必须计算两类误差：

#### A. 原始边缘到矢量边缘

表示原图有线，但矢量没有覆盖到。

```text
missing_edge_error
```

#### B. 矢量边缘到原始边缘

表示矢量画多了、鼓包了、偏出去了。

```text
overdraw_error
```

### 20.2 Chamfer Distance

```text
chamfer_error =
    mean(distance(original_edge_points → vector_edge_points))
  + mean(distance(vector_edge_points → original_edge_points))
```

---

## 21. 复杂度惩罚与过拟合控制

### 21.1 类 BIC 复杂度惩罚

```text
score =
    fit_error
  + λ * model_complexity * log(n)
```

model_complexity：

```text
line      = 2
circle    = 3
arc       = 5
ellipse   = 5
bezier    = 8
bspline   = control_points * 2
polyline  = points * 2
```

### 21.2 非线性控制点惩罚

```text
control_point_penalty = λ * control_point_count ^ 1.5
```

目的：

一个误差略大的圆，应该优于一个误差稍小但有大量控制点的 Bezier。

---

## 22. Scorer 综合评分系统

### 22.1 总分

```text
total_score =
    edge_error_score
  + geometry_complexity_score
  + smoothness_score
  + constraint_violation_score
  + shared_tangent_violation_score
  + semantic_penalty_score
  + topology_error_score
  + self_intersection_score
  + color_error_score
  + alpha_sampling_confidence_score
  + coordinate_consistency_score
```

分数越低越好。

---

### 22.2 topology_error_score

```text
topology_error =
    gap_count * w1
  + max_gap * w2
  + self_intersection_count * w3
```

自交应作为严重错误处理：

```text
self_intersection_score = self_intersection_count * high_weight
```

---

### 22.3 shared_tangent_violation_score

```text
shared_tangent_violation =
    angle_between(segment_a_tangent, shared_tangent)
  + angle_between(segment_b_tangent, shared_tangent)
```

用于评估 G1 共切是否真正成立。

---

### 22.4 constraint_violation_score

```text
constraint_violation =
    horizontal_error
  + vertical_error
  + tangent_error
  + concentric_error
  + coincident_error
  + symmetry_error
  + g1_error
```

---

## 23. AI Agent 设计

### 23.1 AI 的角色

AI 不直接拟合曲线，而是做：

1. 视觉审查
2. 语义判断
3. 问题定位
4. 修改意图规划
5. 约束建议
6. 过拟合识别
7. 批量修改建议
8. 根据 RefinementFeedback 重新评估错误提议
9. 根据 self_intersection 反馈建议回滚或重切分
10. 根据 alpha / color variance 建议样式处理

---

### 23.2 AI 输入

输入包括：

1. 原始图像
2. 当前 overlay 图
3. Distance Field Diff 图
4. 当前 VectorDocument JSON
5. 每段 fit_error
6. 每段 complexity_score
7. 每段 constraint violation
8. topology_status
9. self_intersection_count
10. coordinate_system
11. 用户锁定信息
12. RefinementFeedback
13. alpha / color variance
14. 当前可用工具列表

---

## 24. AI Prompt 模板

```text
你是一个矢量重建审查器。

你的任务：
1. 对比原始图像、当前矢量叠加图和距离场差异图。
2. 找出不符合人工临摹习惯的地方。
3. 优先使用标准几何：直线、圆、圆弧、椭圆。
4. 避免用大量 Bezier 或 B 样条拟合标准几何。
5. 减少控制点数量。
6. 保持几何约束：水平、垂直、同心、相切、重合、G1 连续。
7. 不修改用户锁定的路径、分段、控制点或约束。
8. 只输出修改意图，不直接输出精确几何参数。
9. 如果发现疑似噪声或断裂，请建议使用鲁棒精化，而不是强行拟合。
10. 如果发现路径闭合缝隙，请建议执行拓扑闭合修正。
11. 如果发现路径自交，请建议回滚、重新切分或改用更简单的几何表达。
12. 如果算法反馈某个提议 inlier_ratio 过低，请重新评估该区域是否应改为 Bezier、噪声或重新切分。
13. 如果颜色方差过高，可以建议进行渐变检测，但不要强制设为纯色。
14. 如果 alpha_variance 过高，请标记 transparency_candidate。
15. 如果多个修改相互独立，可以使用 propose_batch_refinement 批量提交修改意图。

输出 JSON：
{
  "summary": "",
  "issues": [],
  "proposed_commands": []
}
```

---

## 25. AI 工具协议

### 25.1 查询类工具

```json
{"tool": "list_objects"}
```

```json
{"tool": "list_paths"}
```

```json
{"tool": "get_path", "path_id": "path_001"}
```

```json
{"tool": "get_segment", "segment_id": "seg_001"}
```

```json
{"tool": "get_constraints", "target_id": "path_001"}
```

```json
{"tool": "get_refinement_feedback", "proposal_id": "proposal_001"}
```

---

### 25.2 渲染类工具

```json
{"tool": "render_overlay"}
```

```json
{"tool": "render_distance_field_diff"}
```

```json
{"tool": "render_zoom", "bbox": [100, 120, 260, 240]}
```

---

### 25.3 语义提议类工具

```json
{
  "tool": "propose_replace_path_with_circle",
  "path_id": "path_002"
}
```

```json
{
  "tool": "propose_replace_segment_with_arc",
  "path_id": "path_001",
  "segment_range": [3, 6]
}
```

```json
{
  "tool": "propose_replace_segment_with_line",
  "segment_id": "seg_004"
}
```

```json
{
  "tool": "propose_replace_path_with_ellipse",
  "path_id": "path_005"
}
```

```json
{
  "tool": "propose_gradient_detection",
  "path_id": "path_001"
}
```

---

### 25.4 批量提议工具

```json
{
  "tool": "propose_batch_refinement",
  "commands": [
    {
      "tool": "propose_replace_segment_with_arc",
      "path_id": "path_001",
      "segment_range": [3, 6],
      "confidence": 0.88
    },
    {
      "tool": "propose_replace_segment_with_line",
      "segment_id": "seg_009",
      "confidence": 0.93
    }
  ]
}
```

---

### 25.5 约束命令

```json
{
  "tool": "apply_coincident_constraint",
  "targets": ["anchor_001", "anchor_087"]
}
```

```json
{
  "tool": "apply_shared_tangent_constraint",
  "anchor_id": "anchor_008",
  "segments": ["seg_001", "seg_002"]
}
```

```json
{
  "tool": "apply_tangent_constraint",
  "targets": ["seg_003", "seg_004"]
}
```

```json
{
  "tool": "apply_concentric_constraint",
  "targets": ["circle_001", "circle_002"]
}
```

---

### 25.6 拓扑命令

```json
{
  "tool": "enforce_path_topology",
  "path_id": "path_001"
}
```

```json
{
  "tool": "detect_self_intersection",
  "path_id": "path_001"
}
```

---

### 25.7 锁定命令

```json
{
  "tool": "lock_segment",
  "segment_id": "seg_005",
  "lock_type": "geometry"
}
```

---

## 26. CommandExecutor 命令执行器

### 26.1 职责

1. 校验 AI 命令是否合法。
2. 检查用户锁定。
3. 检查坐标系统。
4. 调用 BreakPointOptimizer。
5. 调用 RefinementEngine。
6. 调用 SharedTangentOptimizer。
7. 获取 RefinementFeedback。
8. 应用 SegmentRigidityPolicy。
9. 应用 TopologyEngine。
10. 检测 self-intersection。
11. 应用 ConstraintGraph。
12. 更新 VectorDocument。
13. 重新评分。
14. 记录历史快照。
15. 返回执行结果和影响范围。

---

### 26.2 执行结果

```json
{
  "success": true,
  "command_id": "cmd_001",
  "affected_paths": ["path_001"],
  "affected_segments": ["seg_003", "seg_004"],
  "old_score": 12.4,
  "new_score": 8.7,
  "topology_status": "closed",
  "self_intersection_count": 0,
  "requires_rerender": true
}
```

---

## 27. BatchCommandExecutor 批量命令执行器

默认策略：

```text
逐条执行
失败跳过
不回滚整个 batch
每条命令都返回独立反馈
```

可选策略：

```text
continue_on_failure = true
rollback_batch_on_failure = false
```

---

### 27.1 执行结果

```json
{
  "batch_id": "batch_001",
  "success_count": 2,
  "failure_count": 1,
  "results": [
    {
      "command_index": 0,
      "success": true,
      "old_score": 12.3,
      "new_score": 9.1
    },
    {
      "command_index": 1,
      "success": false,
      "reason": "low_inlier_ratio",
      "inlier_ratio": 0.38
    },
    {
      "command_index": 2,
      "success": true,
      "old_score": 9.1,
      "new_score": 8.4
    }
  ]
}
```

---

## 28. HistoryManager 回滚机制

每次执行命令前保存快照。

```json
{
  "history_item": {
    "version": 12,
    "command": {},
    "before": {},
    "after": {},
    "old_score": 12.4,
    "new_score": 8.7,
    "timestamp": "2026-05-08T12:00:00"
  }
}
```

支持：

1. undo
2. redo
3. 对比修改前后
4. 回滚 AI 修改
5. 保留用户手动修改历史
6. 支持 batch 级回滚和单条命令级回滚

---

## 29. 用户交互设计

### 29.1 必须支持

1. 选择图片
2. 自动拟合
3. 选择轮廓
4. 淡黄色显示原始轮廓
5. 叠加拟合曲线
6. 显示控制点
7. 拖动控制点
8. 右键辅助分段
9. 指定某段类型
10. 锁定某段
11. 添加约束
12. 查看 coincident 约束
13. 查看 shared tangent 约束
14. 查看 self-intersection 标记
15. AI 审查
16. 应用 AI 建议
17. 批量应用 AI 建议
18. 撤销
19. 导出 SVG / DXF / JSON

---

### 29.2 图层开关

```text
[x] 原图
[x] 原始轮廓
[x] 拟合曲线
[x] 控制点
[ ] 距离场差异图
[x] AI 建议
[x] 分段编号
[x] 几何类型颜色
[ ] 约束标记
[ ] coincident 锚点标记
[ ] shared tangent 标记
[ ] 拓扑错误标记
[ ] 自交点标记
[ ] 切分点标记
```

---

## 30. MVP 实现范围

### 30.1 P0：立即做

1. 定义 dataclass 数据结构。
2. VectorDocument 保持纯数据结构，不包含渲染、拟合、UI 逻辑。
3. 实现 CoordinateTransformer。
4. ContourExtractor 后立即 pixel_to_vector。
5. 内部统一 float Vector Space 坐标。
6. 支持 Document to_json / from_json。
7. 保留 binary_contours 和 skeleton_contours。
8. 增加自适应重采样。
9. 增加 SegmentRigidityPolicy。
10. 增加 BreakPointOptimizer。
11. 增加 GlobalSnappingEngine。
12. 增加 coincident constraint。
13. 增加 Distance Field Diff。
14. 增加 AI 提议 + 算法精化机制。
15. 增加 RefinementFeedback。
16. 增加 RANSAC 圆 / 直线 / 圆弧精化。
17. 增加 Path Closing 拓扑修正。
18. 增加 SelfIntersectionDetector。
19. 增加非线性复杂度惩罚。
20. 增加用户锁定段。
21. 增加基础约束：
    - horizontal
    - vertical
    - concentric
    - tangent
    - coincident
22. 增加 JSON 导出当前矢量结构。
23. 增加 overlay 图导出。
24. 增加 AI 审查按钮。

---

### 30.2 P1：第二阶段

1. G1 continuity candidate 自动检测。
2. SharedTangentOptimizer。
3. Object 层语义合并。
4. 局部约束校正。
5. AlphaAwareStyleAnalyzer 纯色采样。
6. color_variance。
7. alpha_variance。
8. transparency_candidate。
9. 约束违规评分。
10. shared_tangent_violation_score。
11. AI 命令执行器。
12. BatchCommandExecutor。
13. 回滚历史。
14. AI 建议可视化。

---

### 30.3 P2：后续阶段

1. 完整 Constraints Solver。
2. G2 连续性。
3. 复杂布尔运算。
4. 多对象对称识别。
5. 自动多轮 AI 优化。
6. gradient detection。
7. 线性/径向渐变拟合。
8. 从零 AI 临摹复杂图形。

---

## 31. Benchmark 测试集

建议用例：

1. case_001：完整圆
2. case_002：同心圆
3. case_003：圆弧 + 直线
4. case_004：椭圆
5. case_005：机械外轮廓
6. case_006：Logo
7. case_007：文字轮廓
8. case_008：含孔洞闭合图形
9. case_009：对称结构
10. case_010：噪声较多的扫描图
11. case_011：断裂轮廓
12. case_012：带颜色 Logo
13. case_013：混合路径 Line + Arc + Bezier
14. case_014：DXF 单位导出测试
15. case_015：渐变候选 Logo
16. case_016：跨 Path 端点吸附
17. case_017：批量 AI 修改建议
18. case_018：路径自交检测
19. case_019：Line + Arc G1 共切
20. case_020：透明 PNG 图标

每次算法更新后统计：

1. 总误差
2. 控制点数量
3. 圆识别准确率
4. 直线识别准确率
5. 圆弧识别准确率
6. 约束违规数
7. coincident 约束命中率
8. shared tangent 误差
9. 拓扑错误数
10. 自交检测准确率
11. 用户需要手动修正次数
12. 导出 SVG 节点数量
13. 颜色采样准确率
14. alpha 采样准确率
15. DXF 尺寸误差
16. RANSAC 内点率
17. AI 提议失败率
18. batch 命令成功率

---

## 32. 开发路线图

### 阶段 1：数据结构与坐标系统

目标：

把当前 all_segments 升级成纯数据型 VectorDocument，并统一坐标系统。

任务：

1. 定义 core/types.py。
2. 实现 dataclass：
   - Anchor
   - Segment
   - Path
   - Object
   - Constraint
   - Style
   - CoordinateSystem
   - VectorDocument
3. 实现 CoordinateTransformer。
4. 确保 ContourExtractor 后立即 pixel_to_vector。
5. 实现 Document to_json / from_json。
6. 编写 coordinate 和 json 序列化单元测试。

---

### 阶段 2：最小 Pipeline

任务：

1. OpenCV 提取轮廓。
2. Transformer 转换坐标。
3. 简单 line / bezier 分段。
4. 存入 VectorDocument。
5. 导出 JSON。
6. 验证 JSON 可反序列化。

---

### 阶段 3：增强渲染和评分

任务：

1. overlay 渲染
2. distance field diff 渲染
3. edge error
4. complexity score
5. topology score
6. self_intersection_score
7. constraint violation score
8. coordinate consistency score

---

### 阶段 4：鲁棒精化和切分点优化

任务：

1. BreakPointOptimizer
2. RANSAC line
3. RANSAC circle
4. RANSAC arc
5. 离群点剔除
6. 内点重拟合
7. RefinementFeedback

---

### 阶段 5：全局锚点吸附和拓扑检测

任务：

1. GlobalSnappingEngine
2. anchor 空间索引
3. epsilon 搜索
4. coincident constraint 候选生成
5. Path Closing
6. SelfIntersectionDetector
7. 自交点可视化

---

### 阶段 6：AI 审查器

任务：

1. 导出当前 overlay
2. 导出 diff
3. 导出 VectorDocument JSON
4. 调用 AI
5. 显示 AI issues
6. 支持 propose_batch_refinement
7. 暂不自动执行修改

---

### 阶段 7：AI 命令执行器

任务：

1. 定义 command schema
2. 校验命令
3. 检查锁定
4. 优化切分点
5. 调用 RefinementEngine
6. 应用拓扑修正
7. 应用刚性策略
8. 检测自交
9. 应用约束修正
10. 支持 batch 逐条执行
11. 记录历史
12. 重新评分

---

### 阶段 8：共享切线与样式语义

任务：

1. SharedTangentOptimizer
2. shared_tangent_violation_score
3. AlphaAwareStyleAnalyzer
4. fill_color
5. fill_alpha
6. color_variance
7. alpha_variance
8. paint_type
9. SVG 样式导出

---

## 33. 第一版 UI 建议

右侧面板：

```text
[选择图片]
[自动拟合]
[AI 审查]
[应用 AI 建议]
[批量应用 AI 建议]
[自动优化 3 轮]
[撤销]
[导出 SVG]
[导出 DXF]
[导出 JSON]

图层：
[x] 原图
[x] 原始轮廓
[x] 拟合曲线
[x] 控制点
[ ] 距离场差异图
[x] AI 建议
[x] 分段编号
[ ] 约束标记
[ ] coincident 锚点
[ ] shared tangent
[ ] 拓扑错误
[ ] 自交点
[ ] 切分点

评分：
边缘误差：1.23
复杂度：12.4
约束违规：2
拓扑错误：0
自交数量：0
Coincident 约束：5
Shared Tangent 误差：1.4°
RANSAC 内点率：0.86
总评分：18.7
```

---

## 34. 最小可落地版本

第一版只实现：

1. dataclass 数据结构。
2. VectorDocument 纯数据层。
3. CoordinateTransformer。
4. Document to_json / from_json。
5. OpenCV 提取轮廓。
6. pixel_to_vector。
7. 简单分段。
8. 存入 Document。
9. 导出 JSON。
10. 当前 overlay 图导出。
11. Distance Field Diff 图导出。
12. AI 审查 Prompt。
13. AI 返回 issues。
14. UI 展示 AI 建议。

这一步不需要 AI 自动修改路径，但已经可以验证系统底座是否正确。

---

## 35. 第二个可落地版本

在第一版基础上增加：

1. propose_replace_path_with_circle
2. propose_replace_segment_with_line
3. propose_replace_segment_with_arc
4. propose_replace_path_with_ellipse
5. propose_batch_refinement
6. BreakPointOptimizer
7. RANSAC 精化
8. RefinementFeedback
9. GlobalSnappingEngine
10. apply_coincident_constraint
11. apply_shared_tangent_constraint
12. apply_concentric_constraint
13. apply_tangent_constraint
14. set_continuity
15. enforce_path_topology
16. detect_self_intersection
17. SegmentRigidityPolicy
18. lock_segment
19. undo

AI 输出 proposed_commands，用户点击应用后由算法精化执行。

---

## 36. 最终形态

最终系统工作方式：

1. 用户导入图片。
2. 系统自动生成初始矢量。
3. 用户选中某个轮廓。
4. 系统显示淡黄色原始轮廓和拟合曲线。
5. 用户点击 AI 审查。
6. AI 标出：
   - 哪里应为圆
   - 哪里应为圆弧
   - 哪里应为直线
   - 哪里控制点过多
   - 哪里有鼓包
   - 哪里存在噪声
   - 哪里应添加约束
   - 哪里有拓扑缝隙
   - 哪里存在自交
   - 哪里切分点不合理
   - 哪些锚点应重合
   - 哪些连接点应共享切线
   - 哪里颜色可能是渐变
   - 哪里存在透明度混合
7. 用户点击应用。
8. 系统优化切分点。
9. 系统用 RANSAC 和最小二乘重新精确拟合。
10. SharedTangentOptimizer 处理共切连接。
11. RefinementFeedback 判断提议是否有效。
12. GlobalSnappingEngine 推断跨路径 coincident 约束。
13. TopologyEngine 根据刚性策略修正路径闭合。
14. SelfIntersectionDetector 检查自交。
15. ConstraintGraph 修正几何关系。
16. Scorer 验证修改收益。
17. 用户局部拖点或锁定。
18. 最终导出 SVG / DXF / JSON。

---

## 37. 最重要结论

这个系统不应被设计成“AI 直接拟合轮廓”。

正确设计是：

AI 像人工设计师一样理解图形结构，提出修改意图；BreakPointOptimizer 优化物理切分点；RANSAC 和最小二乘根据这些意图精确计算几何参数；SharedTangentOptimizer 保证 G1 共切连接的同步优化；RefinementFeedback 将算法结果反哺 AI；ConstraintGraph 保持整体几何一致性；GlobalSnappingEngine 推断跨路径锚点重合关系；TopologyEngine 基于 SegmentRigidityPolicy 保证路径闭合和混合路径连接；SelfIntersectionDetector 防止路径自交；CoordinateTransformer 保证显示、计算和导出坐标一致；VectorDocument 保持纯数据化；AlphaAwareStyleAnalyzer 降低颜色与透明度采样误差；复杂度惩罚防止过拟合；Distance Field Diff 提供客观评分；用户锁定和回滚保证可控性。

最终核心公式：

视觉理解
+ 单向坐标转换
+ 纯数据文档模型
+ 切分点优化
+ 鲁棒几何拟合
+ 共享切线优化
+ 确定性反馈
+ 全局锚点吸附
+ 约束校正
+ 拓扑维护
+ 自交检测
+ Alpha 感知样式分析
+ 批量提议执行
+ 复杂度惩罚
+ 人工可控
= 高质量矢量重建
