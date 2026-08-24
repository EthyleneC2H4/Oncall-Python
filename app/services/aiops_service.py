"""通用 Plan-Execute-Replan 服务 — PlanExecuteRuntime 的薄门面

图构建与流式执行已迁移至 app.agent.runtime.plan_execute_runtime；
本模块保留旧的 AIOpsService 接口（/api/aiops 与既有测试依赖），
内部委托运行时产出统一的 AgentEvent 流。

SSE 旧事件 dict 契约由 API 层翻译器负责（app/api/event_translator.py，
golden 快照测试钉死）。
"""

from collections.abc import AsyncGenerator
from textwrap import dedent

from app.agent.runtime import AgentEvent, PlanExecuteRuntime


class AIOpsService:
    """通用 Plan-Execute-Replan 服务（薄门面）"""

    def __init__(self, runtime: PlanExecuteRuntime | None = None):
        """初始化服务

        Args:
            runtime: 可注入的 Plan-Execute 运行时（测试用），默认新建
        """
        self.runtime = runtime or PlanExecuteRuntime()

    @property
    def graph(self):
        """底层 LangGraph 编译实例（兼容旧访问路径）"""
        return self.runtime.graph

    async def execute(
        self, user_input: str, session_id: str = "default"
    ) -> AsyncGenerator[AgentEvent, None]:
        """执行 Plan-Execute-Replan 流程，增量产出结构化事件

        Args:
            user_input: 用户的任务描述
            session_id: 会话ID

        Yields:
            AgentEvent: 统一事件流（PLAN_CREATED / STEP_END / REPLAN / REPORT / COMPLETE|ERROR）
        """
        async for event in self.runtime.run(user_input, session_id=session_id):
            yield event

    async def diagnose(self, session_id: str = "default") -> AsyncGenerator[AgentEvent, None]:
        """AIOps 诊断接口（兼容旧接口）

        使用固定的 AIOps 任务描述驱动完整诊断流程。
        complete 事件的 diagnosis 包装由 API 层翻译器完成（diagnosis_mode=True）。

        Args:
            session_id: 会话ID

        Yields:
            AgentEvent: 统一事件流
        """
        aiops_task = self._build_diagnosis_task()
        async for event in self.execute(aiops_task, session_id=session_id):
            yield event

    @staticmethod
    def _build_diagnosis_task() -> str:
        """构建 AIOps 诊断任务提示词"""
        return dedent(
            """诊断当前系统是否存在告警，如果存在告警请详细分析告警原因并生成诊断报告，诊断报告输出格式要求：
                ```
                # 告警分析报告

                ---

                ## 📋 活跃告警清单

                | 告警名称 | 级别 | 目标服务 | 首次触发时间 | 最新触发时间 | 状态 |
                |---------|------|----------|-------------|-------------|------|
                | [告警1名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |
                | [告警2名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |

                ---

                ## 🔍 告警根因分析1 - [告警名称]

                ### 告警详情
                - **告警级别**: [级别]
                - **受影响服务**: [服务名]
                - **持续时间**: [X分钟]

                ### 症状描述
                [根据监控指标描述症状]

                ### 日志证据
                [引用查询到的关键日志]

                ### 根因结论
                [基于证据得出的根本原因]

                ---

                ## 🛠️ 处理方案执行1 - [告警名称]

                ### 已执行的排查步骤
                1. [步骤1]
                2. [步骤2]

                ### 处理建议
                [给出具体的处理建议]

                ### 预期效果
                [说明预期的效果]

                ---

                ## 🔍 告警根因分析2 - [告警名称]
                [如果有第2个告警，重复上述格式]

                ---

                ## 📊 结论

                ### 整体评估
                [总结所有告警的整体情况]

                ### 关键发现
                - [发现1]
                - [发现2]

                ### 后续建议
                1. [建议1]
                2. [建议2]

                ### 风险评估
                [评估当前风险等级和影响范围]
                ```

                **重要提醒**：
                - 最终输出必须是纯 Markdown 文本，不要包含 JSON 结构
                - 所有内容必须基于工具查询的真实数据，严禁编造
                - 如果某个步骤失败，在结论中如实说明，不要跳过"""
        )


# 全局单例
aiops_service = AIOpsService()
