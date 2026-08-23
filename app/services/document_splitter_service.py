"""文档分割服务模块 — 结构感知 + Parent-Child 双层分块

核心设计：
1. 结构感知：按 Markdown 标题层级（H1/H2/H3）切分，代码块和列表作为原子单元不拆分
2. Parent-Child 双层索引：
   - Parent：完整的 H2/H3 章节（保留上下文完整性）
   - Child：Parent 内进一步切分的小片段（200-300 字符，提升检索精度）
3. 检索时用 Child 做精确匹配，返回对应的 Parent 给 LLM
"""

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger

from app.config import config


@dataclass
class Section:
    """一个结构化的文档章节（Parent 单元）"""

    title: str = ""
    level: int = 0  # 标题层级 1/2/3
    content: str = ""  # 完整的章节文本（含标题）
    h1: str = ""
    h2: str = ""
    h3: str = ""


class DocumentSplitterService:
    """结构感知 + Parent-Child 双层分块服务"""

    # 代码块正则（匹配 ```...``` 块）
    _CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    # 标题正则
    _HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    def __init__(self):
        self.child_chunk_size = config.chunk_max_size  # child 目标大小（默认 800 字符）
        self.child_overlap = config.chunk_overlap  # child 重叠（默认 100）
        self.min_child_size = 100  # 小于此大小的 child 与相邻合并
        logger.info(
            f"文档分割服务初始化 (结构感知 + Parent-Child), "
            f"child_size={self.child_chunk_size}, overlap={self.child_overlap}"
        )

    # ── 公开接口 ──────────────────────────────────────────────

    def split_document(self, content: str, file_path: str = "") -> list[Document]:
        """智能分割文档（根据文件类型选择分割器）"""
        if file_path.endswith(".md"):
            return self.split_markdown(content, file_path)
        else:
            return self.split_text(content, file_path)

    def split_markdown(self, content: str, file_path: str = "") -> list[Document]:
        """结构感知 Markdown 分割 → Parent-Child 双层分块

        Returns:
            List[Document]: Child chunks，每个 child 的 metadata 中包含
            parent_id 和 parent_content，用于检索后还原上下文。
        """
        if not content or not content.strip():
            logger.warning(f"Markdown 文档内容为空: {file_path}")
            return []

        try:
            # Step 1: 解析 Markdown 结构 → Parent sections
            sections = self._parse_sections(content)
            logger.info(f"结构解析: {file_path} → {len(sections)} 个 Parent 章节")

            # Step 2: 每个 Parent 切分为 Child chunks
            all_children: list[Document] = []
            for section in sections:
                parent_id = str(uuid.uuid4())
                children = self._split_section_to_children(section, parent_id, file_path)
                all_children.extend(children)

            logger.info(
                f"Markdown 分割完成: {file_path} → "
                f"{len(sections)} parents, {len(all_children)} children"
            )
            return all_children

        except Exception as e:
            logger.error(f"Markdown 分割失败: {file_path}, 错误: {e}")
            raise

    def split_text(self, content: str, file_path: str = "") -> list[Document]:
        """分割纯文本文档（无结构感知，按段落切分）"""
        if not content or not content.strip():
            logger.warning(f"文本文档内容为空: {file_path}")
            return []

        try:
            parent_id = str(uuid.uuid4())
            parent_content = content.strip()

            chunks = self._chunk_text(parent_content, self.child_chunk_size, self.child_overlap)

            docs = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "_source": file_path,
                        "_extension": Path(file_path).suffix,
                        "_file_name": Path(file_path).name,
                        "parent_id": parent_id,
                        "parent_content": parent_content[:6000],  # 防止超大
                        "chunk_index": i,
                    },
                )
                docs.append(doc)

            logger.info(f"文本分割完成: {file_path} → {len(docs)} 个分片")
            return docs

        except Exception as e:
            logger.error(f"文本分割失败: {file_path}, 错误: {e}")
            raise

    # ── Markdown 结构解析 ─────────────────────────────────────

    def _parse_sections(self, content: str) -> list[Section]:
        """将 Markdown 按标题层级解析为 Section 列表

        策略：
        - H1/H2/H3 都作为分割边界
        - 每个 Section 就是一个 Parent chunk
        - 无标题的开头内容归入第一个 Section
        """
        # 保护代码块：用占位符替换，防止代码块中的 # 被误识别为标题
        code_blocks: list[str] = []

        def _replace_code(match: re.Match) -> str:
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        protected = self._CODE_BLOCK_RE.sub(_replace_code, content)

        # 找到所有标题位置
        headings = list(self._HEADING_RE.finditer(protected))

        if not headings:
            # 没有标题，整个文档作为一个 section
            restored = self._restore_code_blocks(protected, code_blocks)
            return [Section(content=restored.strip())]

        sections: list[Section] = []
        current_h1 = ""
        current_h2 = ""

        # 处理第一个标题之前的内容
        if headings[0].start() > 0:
            pre_text = protected[: headings[0].start()].strip()
            if pre_text:
                restored = self._restore_code_blocks(pre_text, code_blocks)
                sections.append(Section(content=restored))

        for i, match in enumerate(headings):
            level = len(match.group(1))  # # = 1, ## = 2, ### = 3
            title = match.group(2).strip()

            # 提取该标题到下一个同级/更高级标题之间的内容
            start = match.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(protected)
            section_text = protected[start:end].strip()

            # 还原代码块
            section_text = self._restore_code_blocks(section_text, code_blocks)

            # 更新标题层级追踪
            if level == 1:
                current_h1 = title
                current_h2 = ""
            elif level == 2:
                current_h2 = title
            # level == 3: h3 = title

            section = Section(
                title=title,
                level=level,
                content=section_text,
                h1=current_h1,
                h2=current_h2 if level >= 2 else "",
                h3=title if level == 3 else "",
            )
            sections.append(section)

        # 合并过小的 section（< 200 字符）到前一个
        merged = self._merge_small_sections(sections, min_size=200)

        return merged

    def _restore_code_blocks(self, text: str, code_blocks: list[str]) -> str:
        """将代码块占位符还原"""
        for i, block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", block)
        return text

    def _merge_small_sections(self, sections: list[Section], min_size: int = 200) -> list[Section]:
        """合并过小的 section 到相邻 section"""
        if len(sections) <= 1:
            return sections

        merged: list[Section] = []
        for section in sections:
            if merged and len(section.content) < min_size and section.level >= merged[-1].level:
                # 合并到前一个 section
                merged[-1].content += "\n\n" + section.content
            else:
                merged.append(section)

        return merged

    # ── Parent → Child 分块 ──────────────────────────────────

    def _split_section_to_children(
        self, section: Section, parent_id: str, file_path: str
    ) -> list[Document]:
        """将一个 Parent section 切分为 Child chunks

        规则：
        1. 代码块永不拆分（作为原子单元）
        2. 按段落 / 列表项切分
        3. 小片段合并，大片段递归切分
        """
        parent_content = section.content.strip()

        if not parent_content:
            return []

        # 按段落和代码块切分为原子块
        atoms = self._split_to_atoms(parent_content)

        # 将原子块组装为 child chunks（目标大小 child_chunk_size）
        chunks = self._assemble_children(atoms)

        # 构建 Document 列表
        docs = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "_source": file_path,
                    "_extension": ".md",
                    "_file_name": Path(file_path).name,
                    "h1": section.h1,
                    "h2": section.h2,
                    "h3": section.h3,
                    "parent_id": parent_id,
                    "parent_content": parent_content[:6000],
                    "chunk_type": self._infer_chunk_type(section.title),
                    "chunk_index": i,
                },
            )
            docs.append(doc)

        return docs

    def _split_to_atoms(self, text: str) -> list[str]:
        """将文本切分为不可再分的原子块

        原子块 = 代码块（整块） | 段落（双换行分隔）
        """
        atoms: list[str] = []

        # 先提取代码块，其余按段落切分
        parts = self._CODE_BLOCK_RE.split(text)
        code_matches = self._CODE_BLOCK_RE.findall(text)
        code_idx = 0

        for _i, part in enumerate(parts):
            # 非代码部分：按双换行切分为段落
            paragraphs = re.split(r"\n\n+", part.strip())
            for para in paragraphs:
                para = para.strip()
                if para:
                    atoms.append(para)

            # 插入对应的代码块（原子，不可再分）
            if code_idx < len(code_matches):
                atoms.append(code_matches[code_idx])
                code_idx += 1

        return atoms

    def _assemble_children(self, atoms: list[str]) -> list[str]:
        """将原子块组装为目标大小的 child chunks

        贪心组装：持续合并原子块，直到超过 child_chunk_size
        """
        if not atoms:
            return []

        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for atom in atoms:
            atom_len = len(atom)

            # 如果单个原子块就超过 child_chunk_size（如大代码块），独立成 chunk
            if atom_len > self.child_chunk_size and not current_parts:
                chunks.append(atom)
                continue

            # 如果加入后会超过大小限制，先保存当前 chunk
            if current_len + atom_len > self.child_chunk_size and current_parts:
                chunks.append("\n\n".join(current_parts))
                # overlap: 保留最后一个 part 作为重叠
                if self.child_overlap > 0 and current_parts:
                    last = current_parts[-1]
                    current_parts = [last] if len(last) < self.child_overlap * 2 else []
                    current_len = sum(len(p) for p in current_parts)
                else:
                    current_parts = []
                    current_len = 0

            current_parts.append(atom)
            current_len += atom_len

        # 处理剩余
        if current_parts:
            remaining = "\n\n".join(current_parts)
            # 如果剩余太小，合并到上一个 chunk
            if chunks and len(remaining) < self.min_child_size:
                chunks[-1] += "\n\n" + remaining
            else:
                chunks.append(remaining)

        return chunks

    def _infer_chunk_type(self, title: str) -> str:
        """根据标题推断 chunk 类型"""
        if not title:
            return "general"
        title_lower = title.lower()
        if any(kw in title_lower for kw in ["步骤", "step", "排查", "操作"]):
            return "procedure"
        if any(kw in title_lower for kw in ["原因", "cause", "分析"]):
            return "root_cause"
        if any(kw in title_lower for kw in ["处理", "方案", "措施", "优化", "action"]):
            return "action"
        if any(kw in title_lower for kw in ["告警", "alert", "描述"]):
            return "alert_info"
        if any(kw in title_lower for kw in ["验证", "verify", "确认"]):
            return "verification"
        return "general"

    # ── 通用文本切分 ─────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """按段落边界切分文本"""
        paragraphs = re.split(r"\n\n+", text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current_len + len(para) > chunk_size and current:
                chunks.append("\n\n".join(current))
                if overlap > 0 and current:
                    last = current[-1]
                    current = [last] if len(last) < overlap * 2 else []
                    current_len = sum(len(p) for p in current)
                else:
                    current = []
                    current_len = 0
            current.append(para)
            current_len += len(para)

        if current:
            chunks.append("\n\n".join(current))

        return chunks if chunks else [text]


# 全局单例
document_splitter_service = DocumentSplitterService()
