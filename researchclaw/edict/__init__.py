"""Edict 门下省审核集成 — 三省六部制 AI 多 Agent 协作架构。

参考：https://github.com/cft0808/edict

核心功能：
- 门下省审核：参照 IEEE Trans 和 CNS 子刊审稿意见模板
- 结构化评分（1-10 分）和修改建议
- 自动触发修正模式（--refine 参数）
- 最多循环 3 轮审核
- 差异报告生成
- 奏折记录与军机处看板流转
"""

from researchclaw.edict.client import EdictClient
from researchclaw.edict.models import MenxiaReview, ReviewCriteria, ReviewResult

__all__ = [
    "EdictClient",
    "MenxiaReview",
    "ReviewCriteria",
    "ReviewResult",
]
