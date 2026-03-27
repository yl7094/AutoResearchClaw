"""Edict 门下省审核数据模型。

参照 IEEE Transactions 和 CNS（Cell/Nature/Science）子刊审稿意见模板。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ReviewAspect(str, Enum):
    """审稿维度枚举。"""

    NOVELTY = "novelty"  # 创新性
    SIGNIFICANCE = "significance"  # 重要性
    METHODOLOGY = "methodology"  # 方法严谨性
    EXPERIMENTS = "experiments"  # 实验充分性
    REPRODUCIBILITY = "reproducibility"  # 可复现性
    CLARITY = "clarity"  # 表述清晰度
    REFERENCES = "references"  # 文献引用质量


class Recommendation(str, Enum):
    """审稿推荐意见。"""

    ACCEPT = "accept"  # 接收
    MINOR_REVISION = "minor_revision"  # 小修
    MAJOR_REVISION = "major_revision"  # 大修
    REJECT = "reject"  # 拒稿


@dataclass
class AspectScore:
    """单个维度的评分。"""

    aspect: ReviewAspect
    score: int  # 1-10 分
    comments: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 1 <= self.score <= 10:
            raise ValueError(f"Score must be between 1 and 10, got {self.score}")


@dataclass
class ReviewCriteria:
    """门下省审核标准配置。

    参照 IEEE Trans 和 CNS 子刊标准：
    - 创新性 >= 7：必须有明确的技术突破或理论贡献
    - 实验充分性 >= 7：必须包含顶会级对比实验
    - 可复现性 >= 7：必须提供代码和数据
    - 文献质量 >= 7：必须引用真实高被引文献（被引>100）
    """

    min_novelty_score: int = 7
    min_experiment_score: int = 7
    min_reproducibility_score: int = 7
    min_reference_score: int = 7
    min_overall_score: float = 7.0
    max_revision_rounds: int = 3

    # 强制性要求
    require_comparison_table: bool = True  # 顶会级实验对比表
    require_reproducibility_statement: bool = True  # 可复现性声明
    require_code_link: bool = True  # 开源代码链接
    require_highly_cited_refs: bool = True  # 高被引文献（被引>100）
    min_highly_cited_count: int = 5  # 最少高被引文献数量


@dataclass
class ReviewResult:
    """门下省审核结果。"""

    overall_score: float  # 总体评分（1-10）
    recommendation: Recommendation
    aspect_scores: list[AspectScore]
    summary: str  # 审稿总结
    major_concerns: list[str]  # 主要问题
    revision_suggestions: list[str]  # 修改建议
    minor_comments: list[str] = field(default_factory=list)  # 次要意见
    confidential_comments: str = ""  # 保密意见（给编辑）
    is_passed: bool = False  # 是否通过审核
    requires_revision: bool = False  # 是否需要修改
    requires_human_intervention: bool = False  # 是否需要人工干预

    def __post_init__(self) -> None:
        self.is_passed = self.overall_score >= 7.0 and self.recommendation in {
            Recommendation.ACCEPT,
            Recommendation.MINOR_REVISION,
        }
        self.requires_revision = self.recommendation in {
            Recommendation.MAJOR_REVISION,
            Recommendation.MINOR_REVISION,
        }


@dataclass
class MenxiaReview:
    """门下省奏折 — 完整审核记录。"""

    review_id: str
    stage_id: int  # ResearchClaw pipeline stage number
    stage_name: str
    paper_title: str
    review_result: ReviewResult
    criteria: ReviewCriteria
    revision_round: int = 0
    previous_scores: list[float] = field(default_factory=list)
    diff_report: str = ""  # 差异报告
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # 军机处看板流转信息
    task_id: str = ""  # Edict task_id
    trace_id: str = ""  # Edict trace_id
    state: str = "Menxia"  # 门下省状态
    flow_log: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典格式。"""
        return {
            "review_id": self.review_id,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "paper_title": self.paper_title,
            "overall_score": self.review_result.overall_score,
            "recommendation": self.review_result.recommendation.value,
            "is_passed": self.review_result.is_passed,
            "requires_revision": self.review_result.requires_revision,
            "requires_human_intervention": self.review_result.requires_human_intervention,
            "revision_round": self.revision_round,
            "previous_scores": self.previous_scores,
            "diff_report": self.diff_report,
            "summary": self.review_result.summary,
            "major_concerns": self.review_result.major_concerns,
            "revision_suggestions": self.review_result.revision_suggestions,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "state": self.state,
            "flow_log": self.flow_log,
        }
