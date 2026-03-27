"""Edict 门下省审核客户端。

集成 Edict 三省六部制 AI 多 Agent 协作架构，实现：
- 门下省审核请求
- IEEE Trans/CNS 子刊标准的结构化评分
- 自动修正循环（最多 3 轮）
- 差异报告生成
- 军机处看板流转记录
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from researchclaw.config import RCConfig
from researchclaw.edict.models import (
    AspectScore,
    MenxiaReview,
    Recommendation,
    ReviewCriteria,
    ReviewResult,
    ReviewAspect,
)
from researchclaw.llm.client import LLMClient

logger = logging.getLogger(__name__)


# IEEE Transactions 和 CNS 子刊审稿意见模板
MENXIA_REVIEW_PROMPT = """You are a senior reviewer for IEEE Transactions and CNS (Cell/Nature/Science) sub-journals.
Please review the following research paper draft according to top-tier academic standards.

## Review Criteria (IEEE Trans / CNS Standards)

### 1. Novelty (创新性) - Weight: 25%
- Does the work present significant technical breakthroughs or theoretical contributions?
- Is the approach fundamentally different from existing methods?
- Score 7+: Clear innovation beyond incremental improvements

### 2. Significance (重要性) - Weight: 15%
- Does this work address an important problem in the field?
- Will the results have broad impact?
- Score 7+: High potential impact on research community

### 3. Methodology Rigor (方法严谨性) - Weight: 20%
- Are the methods sound and well-justified?
- Are assumptions clearly stated and reasonable?
- Score 7+: Rigorous methodology with proper validation

### 4. Experimental Sufficiency (实验充分性) - Weight: 25%
- Must include top-tier conference-level comparison tables
- Are baselines state-of-the-art and fairly compared?
- Are ablation studies comprehensive?
- Score 7+: Extensive experiments with SOTA comparisons

### 5. Reproducibility (可复现性) - Weight: 10%
- Must provide reproducibility statement
- Must include open-source code link
- Are experimental details sufficient for reproduction?
- Score 7+: Full code and data availability

### 6. Clarity (表述清晰度) - Weight: 5%
- Is the paper well-organized and clearly written?
- Are figures and tables effective?
- Score 7+: Professional presentation

### 7. Reference Quality (文献引用质量) - Weight: 5%
- Must cite real highly-cited papers (citations > 100)
- Minimum 5 highly-cited references required
- Are citations relevant and up-to-date?
- Score 7+: Comprehensive citation of seminal works

## Mandatory Requirements (Any failure = automatic major revision)
- [ ] Top-tier experimental comparison table present
- [ ] Reproducibility statement included
- [ ] Open-source code link provided
- [ ] At least 5 highly-cited references (citations > 100)

## Output Format (JSON only, no markdown)

{
    "overall_score": <float 1-10>,
    "recommendation": "<accept|minor_revision|major_revision|reject>",
    "aspect_scores": [
        {"aspect": "novelty", "score": <int 1-10>, "comments": "...", "strengths": [...], "weaknesses": [...]},
        {"aspect": "significance", "score": <int 1-10>, "comments": "...", "strengths": [...], "weaknesses": [...]},
        {"aspect": "methodology", "score": <int 1-10>, "comments": "...", "strengths": [...], "weaknesses": [...]},
        {"aspect": "experiments", "score": <int 1-10>, "comments": "...", "strengths": [...], "weaknesses": [...]},
        {"aspect": "reproducibility", "score": <int 1-10>, "comments": "...", "strengths": [...], "weaknesses": [...]},
        {"aspect": "clarity", "score": <int 1-10>, "comments": "...", "strengths": [...], "weaknesses": [...]},
        {"aspect": "references", "score": <int 1-10>, "comments": "...", "strengths": [...], "weaknesses": [...]}
    ],
    "summary": "<overall summary>",
    "major_concerns": ["<concern 1>", "<concern 2>", ...],
    "revision_suggestions": ["<suggestion 1>", "<suggestion 2>", ...],
    "minor_comments": ["<comment 1>", ...],
    "confidential_comments": "<comments to editor only>",
    "mandatory_checks": {
        "has_comparison_table": <bool>,
        "has_reproducibility_statement": <bool>,
        "has_code_link": <bool>,
        "highly_cited_count": <int>,
        "all_passed": <bool>
    }
}

## Paper to Review

**Title**: {title}

**Abstract**:
{abstract}

**Full Paper Content**:
{paper_content}

Please provide your review strictly in JSON format without any markdown wrapping.
"""


class EdictClient:
    """Edict 门下省审核客户端。

    支持：
    - 本地 Edict 服务 API 调用
    - LLM-based 审核（fallback）
    - 审核结果持久化
    - 军机处看板集成
    """

    def __init__(
        self,
        edict_base_url: str = "http://127.0.0.1:8000",
        llm_client: LLMClient | None = None,
        config: RCConfig | None = None,
    ):
        """初始化 Edict 客户端。

        Args:
            edict_base_url: Edict 后端服务地址
            llm_client: LLM 客户端（用于 fallback 审核）
            config: ResearchClaw 配置
        """
        self.edict_base_url = edict_base_url.rstrip("/")
        self.llm_client = llm_client
        self.config = config
        self._session = requests.Session()
        self._session.timeout = 60

    @classmethod
    def from_rc_config(cls, config: RCConfig) -> "EdictClient":
        """从 ResearchClaw 配置创建客户端。"""
        llm_client = None
        try:
            from researchclaw.llm import create_llm_client
            llm_client = create_llm_client(config)
        except Exception as e:
            logger.warning(f"Failed to create LLM client for Edict: {e}")

        edict_url = getattr(config, "edict", {}).get("base_url", "http://127.0.0.1:8000")
        return cls(edict_base_url=edict_url, llm_client=llm_client, config=config)

    def submit_review_request(
        self,
        stage_id: int,
        stage_name: str,
        paper_title: str,
        paper_content: str,
        abstract: str = "",
        criteria: ReviewCriteria | None = None,
    ) -> MenxiaReview:
        """提交门下省审核请求。

        Args:
            stage_id: ResearchClaw pipeline stage 编号
            stage_name: Stage 名称
            paper_title: 论文标题
            paper_content: 论文全文
            abstract: 摘要
            criteria: 审核标准配置

        Returns:
            MenxiaReview: 门下省审核结果
        """
        review_id = f"mx-{stage_id}-{uuid.uuid4().hex[:8]}"
        criteria = criteria or ReviewCriteria()

        # 尝试调用 Edict 本地服务
        try:
            review_result = self._call_edict_api(
                stage_id=stage_id,
                stage_name=stage_name,
                paper_title=paper_title,
                paper_content=paper_content,
                abstract=abstract,
            )
        except Exception as e:
            logger.warning(f"Edict API call failed, using LLM fallback: {e}")
            review_result = self._llm_review(
                paper_title=paper_title,
                paper_content=paper_content,
                abstract=abstract,
                criteria=criteria,
            )

        # 检查强制性要求
        self._check_mandatory_requirements(review_result, criteria)

        # 创建奏折
        menxia_review = MenxiaReview(
            review_id=review_id,
            stage_id=stage_id,
            stage_name=stage_name,
            paper_title=paper_title,
            review_result=review_result,
            criteria=criteria,
        )

        # 记录到 Edict 任务系统
        self._record_to_edict_task(menxia_review)

        return menxia_review

    def _call_edict_api(
        self,
        stage_id: int,
        stage_name: str,
        paper_title: str,
        paper_content: str,
        abstract: str,
    ) -> ReviewResult:
        """调用 Edict 本地服务 API。"""
        url = f"{self.edict_base_url}/api/v1/menxia/review"
        payload = {
            "stage_id": stage_id,
            "stage_name": stage_name,
            "paper_title": paper_title,
            "paper_content": paper_content[:50000],  # 限制长度
            "abstract": abstract,
        }

        response = self._session.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        return self._parse_review_result(data)

    def _llm_review(
        self,
        paper_title: str,
        paper_content: str,
        abstract: str,
        criteria: ReviewCriteria,
    ) -> ReviewResult:
        """使用 LLM 进行门下省审核（fallback）。"""
        if not self.llm_client:
            raise RuntimeError("No LLM client available for review")

        prompt = MENXIA_REVIEW_PROMPT.format(
            title=paper_title,
            abstract=abstract or paper_content[:1000],
            paper_content=paper_content[:40000],
        )

        response = self.llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.3,
        )

        # 解析 JSON 响应
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse review JSON: {e}")
            # 返回默认的大修结果
            return ReviewResult(
                overall_score=5.0,
                recommendation=Recommendation.MAJOR_REVISION,
                aspect_scores=[],
                summary="Failed to parse review response",
                major_concerns=["Review parsing failed"],
                revision_suggestions=["Please revise and resubmit"],
                requires_human_intervention=True,
            )

        return self._parse_review_result(data)

    def _parse_review_result(self, data: dict[str, Any]) -> ReviewResult:
        """解析审核结果数据。"""
        aspect_scores = []
        for asp in data.get("aspect_scores", []):
            try:
                aspect = ReviewAspect(asp["aspect"])
                aspect_scores.append(
                    AspectScore(
                        aspect=aspect,
                        score=int(asp["score"]),
                        comments=asp.get("comments", ""),
                        strengths=asp.get("strengths", []),
                        weaknesses=asp.get("weaknesses", []),
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning(f"Invalid aspect score: {e}")

        overall_score = float(data.get("overall_score", 5.0))
        rec_str = data.get("recommendation", "major_revision").lower()

        try:
            recommendation = Recommendation(rec_str)
        except ValueError:
            recommendation = Recommendation.MAJOR_REVISION

        # 检查是否需要人工干预
        requires_human = overall_score < 5.0 or recommendation == Recommendation.REJECT

        return ReviewResult(
            overall_score=overall_score,
            recommendation=recommendation,
            aspect_scores=aspect_scores,
            summary=data.get("summary", ""),
            major_concerns=data.get("major_concerns", []),
            revision_suggestions=data.get("revision_suggestions", []),
            minor_comments=data.get("minor_comments", []),
            confidential_comments=data.get("confidential_comments", ""),
            requires_human_intervention=requires_human,
        )

    def _check_mandatory_requirements(
        self,
        result: ReviewResult,
        criteria: ReviewCriteria,
    ) -> None:
        """检查强制性要求。"""
        # 检查各维度最低分
        for aspect_score in result.aspect_scores:
            if aspect_score.aspect == ReviewAspect.NOVELTY:
                if aspect_score.score < criteria.min_novelty_score:
                    result.major_concerns.append(
                        f"Novelty score ({aspect_score.score}) below minimum ({criteria.min_novelty_score})"
                    )
            elif aspect_score.aspect == ReviewAspect.EXPERIMENTS:
                if aspect_score.score < criteria.min_experiment_score:
                    result.major_concerns.append(
                        f"Experiments score ({aspect_score.score}) below minimum ({criteria.min_experiment_score})"
                    )
            elif aspect_score.aspect == ReviewAspect.REPRODUCIBILITY:
                if aspect_score.score < criteria.min_reproducibility_score:
                    result.major_concerns.append(
                        f"Reproducibility score ({aspect_score.score}) below minimum ({criteria.min_reproducibility_score})"
                    )
            elif aspect_score.aspect == ReviewAspect.REFERENCES:
                if aspect_score.score < criteria.min_reference_score:
                    result.major_concerns.append(
                        f"References score ({aspect_score.score}) below minimum ({criteria.min_reference_score})"
                    )

        # 检查总体分数
        if result.overall_score < criteria.min_overall_score:
            result.requires_revision = True
            if result.recommendation == Recommendation.ACCEPT:
                result.recommendation = Recommendation.MINOR_REVISION

    def _record_to_edict_task(self, review: MenxiaReview) -> None:
        """记录审核结果到 Edict 任务系统。"""
        try:
            url = f"{self.edict_base_url}/api/tasks"
            payload = {
                "title": f"门下省审核：{review.paper_title}",
                "description": review.review_result.summary,
                "priority": "高" if review.review_result.requires_revision else "中",
                "assignee_org": "门下省",
                "creator": "researchclaw",
                "tags": ["menxia", "review", f"stage-{review.stage_id}"],
                "meta": review.to_dict(),
            }

            response = self._session.post(url, json=payload, timeout=10)
            if response.status_code == 201:
                task_data = response.json()
                review.task_id = task_data.get("task_id", "")
                review.trace_id = task_data.get("trace_id", "")
                logger.info(f"Recorded review to Edict task: {review.task_id}")
        except Exception as e:
            logger.warning(f"Failed to record to Edict task system: {e}")

    def generate_diff_report(
        self,
        original_content: str,
        revised_content: str,
        revision_suggestions: list[str],
    ) -> str:
        """生成差异报告。"""
        if not self.llm_client:
            return "LLM client not available for diff report generation."

        prompt = f"""Compare the original and revised paper versions.
Original suggestions that needed addressing:
{json.dumps(revision_suggestions, indent=2, ensure_ascii=False)}

Please generate a concise diff report highlighting:
1. What changes were made
2. Which suggestions were addressed
3. What issues remain (if any)

Keep it under 500 words."""

        response = self.llm_client.chat(
            messages=[
                {"role": "system", "content": "You are a technical reviewer."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
        )

        return response.content

    def execute_review_loop(
        self,
        stage_id: int,
        stage_name: str,
        paper_title: str,
        paper_content: str,
        abstract: str = "",
        refine_callback: callable | None = None,
    ) -> MenxiaReview:
        """执行审核 - 修正循环（最多 3 轮）。

        Args:
            stage_id: Stage 编号
            stage_name: Stage 名称
            paper_title: 论文标题
            paper_content: 论文内容
            abstract: 摘要
            refine_callback: 修正回调函数，接收 (suggestions, current_content) 返回修订后内容

        Returns:
            MenxiaReview: 最终审核结果
        """
        criteria = ReviewCriteria()
        current_content = paper_content
        previous_scores: list[float] = []

        for revision_round in range(criteria.max_revision_rounds + 1):
            logger.info(f"Menxia review round {revision_round + 1}/{criteria.max_revision_rounds + 1}")

            # 提交审核
            review = self.submit_review_request(
                stage_id=stage_id,
                stage_name=stage_name,
                paper_title=paper_title,
                paper_content=current_content,
                abstract=abstract,
                criteria=criteria,
            )
            review.revision_round = revision_round
            review.previous_scores = previous_scores.copy()

            # 通过审核
            if review.review_result.is_passed:
                logger.info(f"Paper passed Menxia review with score {review.review_result.overall_score}")
                review.state = "Done"
                self._update_edict_task_state(review, "Done")
                return review

            # 需要人工干预
            if review.review_result.requires_human_intervention:
                logger.warning(f"Review requires human intervention (score: {review.review_result.overall_score})")
                review.diff_report = "Requires manual review due to low quality."
                review.state = "Blocked"
                self._update_edict_task_state(review, "Blocked")
                return review

            # 达到最大轮次
            if revision_round >= criteria.max_revision_rounds:
                logger.warning(f"Max revision rounds reached. Marking for human intervention.")
                review.diff_report = self.generate_diff_report(
                    paper_content, current_content, review.review_result.revision_suggestions
                )
                review.requires_human_intervention = True
                review.state = "NeedsHumanIntervention"
                self._update_edict_task_state(review, "Blocked")
                return review

            # 执行修正
            if refine_callback:
                logger.info(f"Triggering refine mode with {len(review.review_result.revision_suggestions)} suggestions")
                try:
                    revised_content = refine_callback(
                        review.review_result.revision_suggestions,
                        current_content,
                    )
                    # 生成差异报告
                    review.diff_report = self.generate_diff_report(
                        current_content, revised_content, review.review_result.revision_suggestions
                    )
                    current_content = revised_content
                    previous_scores.append(review.review_result.overall_score)
                except Exception as e:
                    logger.error(f"Refine callback failed: {e}")
                    review.requires_human_intervention = True
                    review.state = "RefineFailed"
                    self._update_edict_task_state(review, "Blocked")
                    return review
            else:
                logger.warning("No refine callback provided, cannot auto-revise")
                review.requires_human_intervention = True
                review.state = "NoRefineCallback"
                self._update_edict_task_state(review, "Blocked")
                return review

        # Should not reach here
        return review

    def _update_edict_task_state(self, review: MenxiaReview, new_state: str) -> None:
        """更新 Edict 任务状态。"""
        if not review.task_id:
            return

        try:
            url = f"{self.edict_base_url}/api/tasks/{review.task_id}/transition"
            payload = {
                "new_state": new_state,
                "agent": "menxia",
                "reason": f"Review {'passed' if new_state == 'Done' else 'requires attention'}",
            }
            self._session.post(url, json=payload, timeout=10)
        except Exception as e:
            logger.warning(f"Failed to update Edict task state: {e}")

    def save_review_report(self, review: MenxiaReview, output_dir: Path) -> Path:
        """保存审核报告到文件。"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON 报告
        json_path = output_dir / f"{review.review_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(review.to_dict(), f, indent=2, ensure_ascii=False)

        # Markdown 报告
        md_path = output_dir / f"{review.review_id}.md"
        md_content = self._generate_markdown_report(review)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Saved review reports to {output_dir}")
        return md_path

    def _generate_markdown_report(self, review: MenxiaReview) -> str:
        """生成 Markdown 格式的审核报告。"""
        r = review.review_result

        md = f"""# 门下省审核报告

**审核 ID**: {review.review_id}
**Stage**: {review.stage_id} ({review.stage_name})
**论文标题**: {review.paper_title}
**审核时间**: {review.created_at}
**修订轮次**: {review.revision_round}

## 总体评分

- **总分**: {r.overall_score:.1f}/10.0
- **推荐意见**: {r.recommendation.value}
- **审核状态**: {"✅ 通过" if r.is_passed else "❌ 需修改"}
- **人工干预**: {"⚠️ 是" if r.requires_human_intervention else "否"}

## 维度评分

| 维度 | 评分 | 说明 |
|------|------|------|
"""
        for asp in r.aspect_scores:
            md += f"| {asp.aspect.value} | {asp.score}/10 | {asp.comments[:50]}... |\n"

        md += f"""
## 审稿总结

{r.summary}

## 主要问题

"""
        for i, concern in enumerate(r.major_concerns, 1):
            md += f"{i}. {concern}\n"

        md += """
## 修改建议

"""
        for i, suggestion in enumerate(r.revision_suggestions, 1):
            md += f"{i}. {suggestion}\n"

        if review.diff_report:
            md += f"""
## 差异报告

{review.diff_report}
"""

        if review.previous_scores:
            md += f"""
## 历史评分

"""
            for i, score in enumerate(review.previous_scores, 1):
                md += f"- 第{i}轮：{score:.1f}\n"

        return md
