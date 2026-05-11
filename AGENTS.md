- 当前项目是 AI 临摹式矢量重建系统。
- 所有核心算法必须运行在 Vector Space。
- VectorDocument 必须是纯数据层。
- AI 只输出修改意图，不直接输出精确几何参数。
- 不允许把 UI / OpenCV / AI SDK 引入 core/。
- 每个任务必须跑对应 pytest。
- PR 使用 Refs #issue_number，不使用 Closes，避免绕过 AI Team Board 验收。

Read these files before implementation:
- docs/design/vector-reconstruction.md
- docs/tasks/p0-task-breakdown.md
- docs/process/codex-workflow.md


## Task source

GitHub Issues / AI Dev Team Board are the source of truth for implementation tasks.

Do not infer the task list from local docs.
Do not implement tasks from docs/tasks/*.md unless the user explicitly asks.

Before starting work:
1. Read the assigned GitHub Issue.
2. Treat the Issue body, acceptance criteria, allowed paths, forbidden paths, and test commands as the task contract.
3. Read docs/design/vector-reconstruction.md only for architectural background.
4. Read docs/process/codex-workflow.md for PR and status workflow.

If multiple GitHub tasks are visible, work on exactly one task at a time.
Prefer the lowest-numbered Ready P0 task whose prerequisites are already completed.