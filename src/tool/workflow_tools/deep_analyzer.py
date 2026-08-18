"""提供deep analyzer相关实现。"""

import asyncio
import os
import re
import shutil
import urllib.request
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.message import (
    AudioURL,
    ContentPartAudio,
    ContentPartImage,
    ContentPartPdf,
    ContentPartText,
    ContentPartVideo,
    HumanMessage,
    ImageURL,
    PdfURL,
    SystemMessage,
    VideoURL,
)
from src.model import model_manager
from src.registry import TOOL
from src.tool.default_tools.mdify import MdifyTool
from src.tool.types import Tool, ToolExtra, ToolResponse
from src.tool.workflow_tools.reporter import Report
from src.utils import (
    assemble_project_path,
    dedent,
    fetch_url,
    generate_unique_id,
    get_file_info,
    make_file_url,
    read_lines_file,
)


class FileTypeInfo(BaseModel):
    """定义 `FileTypeInfo`，封装相关数据与行为。"""

    file: str = Field(description="The file path or URL")
    file_type: str = Field(
        description="File type: 'text', 'pdf', 'image', 'audio', or 'video'"
    )


class FileTypeClassification(BaseModel):
    """定义 `FileTypeClassification`，封装相关数据与行为。"""

    files: list[FileTypeInfo] = Field(description="List of files with their types")


class Summary(BaseModel):
    """定义 `Summary`，封装相关数据与行为。"""

    summary: str = Field(
        description="Summary of findings from this chunk (2-3 sentences)"
    )
    found_answer: bool = Field(
        description="Whether the answer to the task has been found in this chunk"
    )
    answer: str | None = Field(
        default=None, description="The answer if found_answer is True, otherwise None"
    )


class SummaryResponse(BaseModel):
    """定义 `SummaryResponse`，封装相关数据与行为。"""

    summary: str = Field(
        description="Summary of findings from this chunk (2-3 sentences)"
    )
    found_answer: bool = Field(
        description="Whether the answer to the task has been found in this chunk"
    )
    answer: str | None = Field(
        default=None, description="The answer if found_answer is True, otherwise None"
    )


_DEEP_ANALYZER_DESCRIPTION = """Deep analysis tool that performs multi-step analysis of complex reasoning tasks with attached files.

🎯 BEST FOR: Complex reasoning tasks that require:
- Multi-step analysis and synthesis
- Integration of information from multiple sources
- Deep understanding of relationships and patterns
- Comprehensive evaluation and conclusion drawing

This tool will:
1. Analyze the provided task and files (text, images, PDFs, Excel, audio, video, etc.)
2. Extract relevant information from files using appropriate methods
3. Perform multimodal analysis preserving visual information from images
4. Perform step-by-step analysis with intelligent approach selection
5. Generate insights and conclusions
6. Continue analysis until answer is found or max steps reached

Supports comprehensive file formats:
• Text & Markup: TXT, MD, JSON, CSV, XML, YAML (supports both local files and URLs like https://example.com/text.txt)
• Programming: PY, JS, HTML, CSS, Java, C/C++ (supports both local files and URLs like https://example.com/code.py)
• Documents: DOCX, XLSX, PPTX (supports both local files and URLs like https://example.com/document.docx)
• Compressed: ZIP, RAR, 7Z, TAR, GZ, BZ2, XZ (supports both local files and URLs like https://example.com/compressed.zip)
• Audio: MP3, WAV, OGG, FLAC, AAC, M4A (supports both local files and URLs like https://example.com/audio.mp3)
• PDF: PDF files (supports both local files and URLs like https://example.com/document.pdf)
• Images: JPG, PNG, GIF, BMP, WebP, TIFF, SVG (multimodal analysis, supports both local files and URLs like https://example.com/image.jpg)
• Video: MP4, AVI, MOV, WMV, WebM or video URL like https://www.youtube.com/watch?v=dQw4w9WgXcQ (supports both local files and URLs, non-YouTube URLs will be downloaded automatically like https://www.youtube.com/watch?v=dQw4w9WgXcQ)

For images, audio, video, preserves visual information by analyzing them directly as message inputs.

💡 Use this tool for complex tasks like:
- Research analysis and synthesis
- Technical document review
- Game strategy analysis (chess, go, etc.)
- Data pattern recognition
- Multi-source information integration
- Complex problem solving requiring multiple perspectives


Args:
- task (str): The task to complete.
- files (Optional[List[str]]): Optional list of absolute file paths or specific URLs (image, video, PDF) to analyze along with the task.

Example: {"name": "deep_analyzer", "args": {"task": "Analyze the given files and provide a summary of the findings.", "files": ["/path/to/file1.txt", "/path/to/file2.pdf"]}}.
"""


@TOOL.register_module(force=True)
class DeepAnalyzerTool(Tool):
    """定义 `DeepAnalyzerTool`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "deep_analyzer"
    description: str = _DEEP_ANALYZER_DESCRIPTION
    metadata: dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(
        default=False, description="Whether the tool requires gradients"
    )

    # 配置相关参数。
    max_rounds: int = Field(
        default=3, description="Maximum analysis rounds in __call__ main loop"
    )
    max_file_size: int = Field(
        default=10 * 1024 * 1024, description="Max file size in bytes (10MB)"
    )
    chunk_size: int = Field(
        default=400, description="Number of lines per chunk for text analysis"
    )
    max_steps: int = Field(
        default=3, description="Maximum steps for image analysis without finding answer"
    )

    model_name: str = Field(
        default="openrouter/gemini-3-flash-preview",
        description="The model to use for the deep analyzer.",
    )
    mdify_tool: MdifyTool = Field(
        default=None, description="The mdify tool to use for the deep analyzer."
    )
    base_dir: str = Field(
        default="workdir/deep_analyzer",
        description="The base directory to use for the deep analyzer.",
    )
    file_model_name: str = Field(
        default="openrouter/gemini-3-flash-preview-plugins",
        description="The model to use for the file analysis.",
    )

    def __init__(
        self,
        model_name: str | None = None,
        base_dir: str | None = None,
        require_grad: bool = False,
        **kwargs,
    ):
        """初始化实例。"""
        super().__init__(require_grad=require_grad, **kwargs)

        if model_name is not None:
            self.model_name = model_name

        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(self.base_dir)

        # 创建所需对象。
        if self.base_dir:
            os.makedirs(self.base_dir, exist_ok=True)

        # 初始化相关状态。
        self.mdify_tool = MdifyTool(base_dir=self.base_dir)

        # 创建所需对象。
        # 处理工具调用。

    def _is_url(self, file_path: str) -> bool:
        """实现 `_is_url` 的业务逻辑。"""
        return file_path.startswith(("http://", "https://"))

    def _is_youtube_url(self, url: str) -> bool:
        """实现 `_is_youtube_url` 的业务逻辑。"""
        youtube_patterns = [
            r"youtube\.com/watch\?v=",
            r"youtu\.be/",
            r"youtube\.com/embed/",
            r"youtube\.com/v/",
        ]
        return any(
            re.search(pattern, url, re.IGNORECASE) for pattern in youtube_patterns
        )

    def _get_url_type(self, url: str) -> str | None:
        """实现 `_get_url_type` 的业务逻辑。"""
        url_lower = url.lower()

        # 校验输入与当前状态。
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".svg",
        ]
        if any(url_lower.endswith(ext) for ext in image_extensions):
            return "image"

        # 校验输入与当前状态。
        if url_lower.endswith(".pdf"):
            return "pdf"

        # 校验输入与当前状态。
        audio_extensions = [
            ".mp3",
            ".wav",
            ".ogg",
            ".flac",
            ".aac",
            ".m4a",
            ".m4b",
            ".m4p",
        ]
        if any(url_lower.endswith(ext) for ext in audio_extensions):
            return "audio"

        # 校验输入与当前状态。
        if self._is_youtube_url(url):
            return "video"

        # 校验输入与当前状态。
        video_extensions = [".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"]
        if any(url_lower.endswith(ext) for ext in video_extensions):
            return "video"

        # 校验输入与当前状态。
        # 说明相关实现细节。
        text_extensions = [
            ".txt",
            ".md",
            ".json",
            ".csv",
            ".xml",
            ".yaml",
            ".yml",
            ".py",
            ".js",
            ".html",
            ".css",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".docx",
            ".doc",
            ".xlsx",
            ".xls",
            ".pptx",
            ".ppt",
            ".zip",
            ".rar",
            ".7z",
            ".tar",
            ".gz",
            ".bz2",
            ".xz",
        ]
        if any(url_lower.endswith(ext) for ext in text_extensions):
            return "text"

        # 加载所需数据。
        # 处理文件与路径。
        return "text"

    async def _download_file(self, url: str, file_type: str = "file") -> str | None:
        """实现 `_download_file` 的业务逻辑。"""
        try:
            # 加载所需数据。
            downloads_dir = os.path.join(self.base_dir, "downloads")
            os.makedirs(downloads_dir, exist_ok=True)

            # 处理文件与路径。
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename or "." not in filename:
                # 处理文件与路径。
                # 处理文件与路径。
                path_ext = os.path.splitext(parsed_url.path)[1]
                if path_ext:
                    default_ext = path_ext
                else:
                    # 处理文件与路径。
                    default_exts = {"video": ".mp4", "audio": ".mp3", "file": ".bin"}
                    default_ext = default_exts.get(file_type, ".bin")
                filename = f"{file_type}_{hash(url) % 100000}{default_ext}"

            local_path = os.path.join(downloads_dir, filename)

            # 加载所需数据。
            logger.info(f"| 📥 Downloading {file_type} from {url} to {local_path}")
            await asyncio.to_thread(urllib.request.urlretrieve, url, local_path)
            logger.info(
                f"| ✅ {file_type.capitalize()} downloaded successfully: {local_path}"
            )
            return local_path
        except Exception as e:
            logger.error(f"| ❌ Error downloading {file_type} from {url}: {e}")
            return None

    async def _classify_files(self, files: list[str]) -> list[FileTypeInfo]:
        """实现 `_classify_files` 的业务逻辑。"""
        try:
            # 校验输入与当前状态。
            path_files = []
            url_classifications = []

            for file_path in files:
                if self._is_url(file_path):
                    url_type = self._get_url_type(file_path)
                    if url_type:
                        url_classifications.append(
                            FileTypeInfo(file=file_path, file_type=url_type)
                        )
                    else:
                        # 说明相关实现细节。
                        logger.warning(f"Unsupported URL type: {file_path}")
                else:
                    path_files.append(file_path)

            # 处理模型调用。
            if path_files:
                # 创建所需对象。
                file_list = "\n".join([f"- {file_path}" for file_path in path_files])

                prompt = dedent(f"""Classify the following files by type. For each file, determine if it is:
                - 'text': Text files, markup files, programming files, documents (DOCX, XLSX, PPTX), or compressed files (ZIP, RAR, 7Z, TAR, GZ, BZ2, XZ)
                - 'pdf': PDF files
                - 'image': Image files (JPG, PNG, GIF, BMP, WebP, TIFF, SVG)
                - 'audio': Audio files (MP3, WAV, OGG, FLAC, AAC, M4A)
                - 'video': Video files (MP4, AVI, MOV, WMV, WebM)

                Files to classify:
                {file_list}

                Classify each file based on its content type, not just the extension.
                """)

                messages = [
                    SystemMessage(
                        content="You are an expert at classifying file types based on their content and purpose."
                    ),
                    HumanMessage(content=prompt),
                ]

                response = await model_manager(
                    model=self.model_name,
                    messages=messages,
                    response_format=FileTypeClassification,
                )

                if not response.success:
                    logger.warning(
                        f"LLM classification failed: {response.message}, using file extension fallback"
                    )
                    path_classifications = self._classify_by_extension(path_files)
                elif response.extra and response.extra.parsed_model:
                    classification = response.extra.parsed_model
                    path_classifications = classification.files
                else:
                    # 执行回退或重试逻辑。
                    logger.warning(
                        "LLM classification failed to parse response, using file extension fallback"
                    )
                    path_classifications = self._classify_by_extension(path_files)
            else:
                path_classifications = []

            # 处理文件与路径。
            return url_classifications + path_classifications

        except Exception as e:
            logger.warning(
                f"Error classifying files with LLM: {e}, using extension fallback"
            )
            # 执行回退或重试逻辑。
            all_classifications = []
            for file_path in files:
                if self._is_url(file_path):
                    url_type = self._get_url_type(file_path)
                    if url_type:
                        all_classifications.append(
                            FileTypeInfo(file=file_path, file_type=url_type)
                        )
                else:
                    # 说明相关实现细节。
                    _, ext = os.path.splitext(file_path.lower())
                    text_exts = [
                        ".txt",
                        ".md",
                        ".json",
                        ".csv",
                        ".xml",
                        ".yaml",
                        ".yml",
                        ".py",
                        ".js",
                        ".html",
                        ".css",
                        ".java",
                        ".cpp",
                        ".c",
                        ".h",
                        ".docx",
                        ".doc",
                        ".xlsx",
                        ".xls",
                        ".pptx",
                        ".ppt",
                        ".zip",
                        ".rar",
                        ".7z",
                        ".tar",
                        ".gz",
                        ".bz2",
                        ".xz",
                    ]
                    pdf_exts = [".pdf"]
                    image_exts = [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".gif",
                        ".bmp",
                        ".webp",
                        ".tiff",
                        ".svg",
                    ]
                    audio_exts = [
                        ".mp3",
                        ".wav",
                        ".ogg",
                        ".flac",
                        ".aac",
                        ".m4a",
                        ".m4b",
                        ".m4p",
                    ]
                    video_exts = [
                        ".mp4",
                        ".avi",
                        ".mov",
                        ".wmv",
                        ".flv",
                        ".webm",
                        ".m4v",
                    ]

                    if ext in text_exts:
                        file_type = "text"
                    elif ext in pdf_exts:
                        file_type = "pdf"
                    elif ext in image_exts:
                        file_type = "image"
                    elif ext in audio_exts:
                        file_type = "audio"
                    elif ext in video_exts:
                        file_type = "video"
                    else:
                        file_type = "text"

                    all_classifications.append(
                        FileTypeInfo(file=file_path, file_type=file_type)
                    )

            return all_classifications

    def _classify_by_extension(self, files: list[str]) -> list[FileTypeInfo]:
        """实现 `_classify_by_extension` 的业务逻辑。"""
        result = []
        for file_path in files:
            # 校验输入与当前状态。
            if self._is_url(file_path):
                url_type = self._get_url_type(file_path)
                if url_type:
                    result.append(FileTypeInfo(file=file_path, file_type=url_type))
                else:
                    logger.warning(f"Unsupported URL: {file_path}, defaulting to text")
                    result.append(FileTypeInfo(file=file_path, file_type="text"))
            else:
                # 处理文件与路径。
                _, ext = os.path.splitext(file_path.lower())

                # 说明相关实现细节。
                text_exts = [
                    ".txt",
                    ".md",
                    ".json",
                    ".csv",
                    ".xml",
                    ".yaml",
                    ".yml",
                    ".py",
                    ".js",
                    ".html",
                    ".css",
                    ".java",
                    ".cpp",
                    ".c",
                    ".h",
                    ".docx",
                    ".doc",
                    ".xlsx",
                    ".xls",
                    ".pptx",
                    ".ppt",
                    ".zip",
                    ".rar",
                    ".7z",
                    ".tar",
                    ".gz",
                    ".bz2",
                    ".xz",
                ]
                pdf_exts = [".pdf"]
                image_exts = [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".gif",
                    ".bmp",
                    ".webp",
                    ".tiff",
                    ".svg",
                ]
                audio_exts = [
                    ".mp3",
                    ".wav",
                    ".ogg",
                    ".flac",
                    ".aac",
                    ".m4a",
                    ".m4b",
                    ".m4p",
                ]
                video_exts = [".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"]

                if ext in text_exts:
                    file_type = "text"
                elif ext in pdf_exts:
                    file_type = "pdf"
                elif ext in image_exts:
                    file_type = "image"
                elif ext in audio_exts:
                    file_type = "audio"
                elif ext in video_exts:
                    file_type = "video"
                else:
                    file_type = "text"  # 说明相关实现细节。

                result.append(FileTypeInfo(file=file_path, file_type=file_type))

        return result

    async def __call__(
        self, task: str, files: list[str] | None = None, **kwargs
    ) -> ToolResponse:
        """执行组件调用并返回结果。"""
        try:
            logger.info(f"| 🚀 Starting DeepAnalyzerTool: {task}")
            if files:
                logger.info(f"| 📂 Attached files: {files}")

            # 说明相关实现细节。
            id = generate_unique_id(prefix="deep_analyzer")

            # 创建所需对象。
            # 创建所需对象。
            md_filename = f"{id}.md"
            report_file_path = (
                os.path.join(self.base_dir, md_filename) if self.base_dir else None
            )

            # 初始化相关状态。
            report = Report(title="Deep Analysis Report", model_name=self.model_name)

            # 转换并规范化数据。
            task_content = f"## Analysis Task\n\n{task}\n\n"
            if files:
                task_content += "## Files\n\n"
                for file in files:
                    file_display = (
                        file if self._is_url(file) else os.path.basename(file)
                    )
                    task_content += f"- {file_display}\n"
                task_content += "\n"

            await report.add_item(content=task_content)

            # 说明相关实现细节。
            summaries: list[Summary] = []

            # 校验输入与当前状态。
            valid_files = []
            if files:
                for file in files:
                    if await self._validate_file(file):
                        valid_files.append(file)
                    else:
                        logger.warning(f"Skipping invalid file: {file}")

            # 处理文件与路径。
            if not valid_files:
                logger.info("| 📝 No files or no valid files, analyzing task directly")
                await self._analyze_task_only(task, summaries, report)

                # 校验输入与当前状态。
                summary = await self._summarize_summaries(task, summaries)
                if summary.found_answer:
                    answer_content = f"## Final Answer\n\n**Answer Found**: Yes\n\n**Answer**: {summary.answer}\n\n"
                    await report.add_item(content=answer_content)

                    if report_file_path:
                        report.report_file_path = report_file_path
                        await report.complete()
                        logger.info(f"✅ Analysis report saved to: {report_file_path}")

                        message = f"Answer found from task analysis.\n\nTask: {task}\n\nAnswer: {summary.answer}, Report saved to: {report_file_path}"

                        return ToolResponse(
                            success=True,
                            message=message,
                            extra=ToolExtra(
                                file_path=report_file_path,
                                data={
                                    "task": task,
                                    "answer_found": True,
                                    "answer": summary.answer,
                                    "file_path": report_file_path,
                                },
                            ),
                        )
                    else:
                        message = f"Answer found from task analysis.\n\nTask: {task}\n\nAnswer: {summary.answer}"
                        return ToolResponse(success=True, message=message)
                else:
                    summaries.append(summary)
                    if report_file_path:
                        report.report_file_path = report_file_path
                        await report.complete()
                        logger.info(f"✅ Analysis report saved to: {report_file_path}")

                        message = (
                            f"Analysis completed but no definitive answer found.\n\nTask: {task}\n\nSummaries:\n"
                            + "\n".join([f"- {s.summary}" for s in summaries])
                            + f"\n\nReport saved to: {report_file_path}"
                        )

                        return ToolResponse(
                            success=False,
                            message=message,
                            extra=ToolExtra(
                                file_path=report_file_path,
                                data={
                                    "task": task,
                                    "answer_found": False,
                                    "file_path": report_file_path,
                                },
                            ),
                        )
                    else:
                        message = (
                            f"Analysis completed but no definitive answer found.\n\nTask: {task}\n\nSummaries:\n"
                            + "\n".join([f"- {s.summary}" for s in summaries])
                        )
                        return ToolResponse(success=False, message=message)

            # 转换并规范化数据。
            logger.info("| 📊 Getting overall file information summary...")
            summary = await self._get_overall_file_summary(task, valid_files)
            if summary and summary.found_answer:
                answer_content = f"## Final Answer\n\n**Answer Found**: Yes\n\n**Answer**: {summary.answer}\n\n"
                await report.add_item(content=answer_content)

                if report_file_path:
                    report.report_file_path = report_file_path
                    await report.complete()
                    logger.info(f"✅ Analysis report saved to: {report_file_path}")

                    message = f"Answer found from file information summary.\n\nTask: {task}\n\nAnswer: {summary.answer}, Report saved to: {report_file_path}"

                    return ToolResponse(
                        success=True,
                        message=message,
                        extra=ToolExtra(
                            file_path=report_file_path,
                            data={
                                "task": task,
                                "answer_found": True,
                                "answer": summary.answer,
                                "file_path": report_file_path,
                            },
                        ),
                    )
                else:
                    message = f"Answer found from file information summary.\n\nTask: {task}\n\nAnswer: {summary.answer}"
                    return ToolResponse(success=True, message=message)
            elif summary:
                summaries.append(summary)
                summary_content = (
                    f"## File Information Summary\n\n{summary.summary}\n\n"
                )
                await report.add_item(content=summary_content)

            # 处理模型调用。
            logger.info(f"| 🔍 Classifying {len(valid_files)} files by type...")
            file_classifications = await self._classify_files(valid_files)

            # 说明相关实现细节。
            classification_content = "## File Classifications\n\n"
            for file_info in file_classifications:
                file_display = (
                    file_info.file
                    if self._is_url(file_info.file)
                    else os.path.basename(file_info.file)
                )
                logger.info(f"| 📋 {file_display}: {file_info.file_type}")
                classification_content += (
                    f"- **{file_display}**: {file_info.file_type}\n"
                )
            classification_content += "\n"
            await report.add_item(content=classification_content)

            # 说明相关实现细节。
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"| 🔄 Main analysis round {round_num}/{self.max_rounds}")

                round_summaries: list[Summary] = []
                round_content = f"## Round {round_num}\n\n"

                # 处理文件与路径。
                for file_info in file_classifications:
                    file = file_info.file
                    file_type = file_info.file_type

                    file_display = (
                        file if self._is_url(file) else os.path.basename(file)
                    )
                    logger.info(f"| 📄 Processing {file_type} file: {file_display}")

                    round_content += (
                        f"### Processing {file_type} file: {file_display}\n\n"
                    )

                    # 处理文件与路径。
                    if file_type == "text":
                        await self._analyze_text_file(task, file, round_summaries)
                    elif file_type == "pdf":
                        await self._analyze_pdf_file(task, file, round_summaries)
                    elif file_type == "image":
                        await self._analyze_image_file(task, file, round_summaries)
                    elif file_type == "audio":
                        await self._analyze_audio_file(task, file, round_summaries)
                    elif file_type == "video":
                        await self._analyze_video_file(task, file, round_summaries)

                    # 处理文件与路径。
                    for s in round_summaries:
                        round_content += f"- {s.summary}\n"
                        if s.found_answer:
                            round_content += f"  **Answer Found**: {s.answer}\n"
                    round_content += "\n"

                    # 校验输入与当前状态。
                    round_summary = await self._summarize_summaries(
                        task, round_summaries
                    )
                    if round_summary.found_answer:
                        round_content += f"### Round {round_num} Summary\n\n**Answer Found**: Yes\n\n**Answer**: {round_summary.answer}\n\n"
                        await report.add_item(content=round_content)

                        if report_file_path:
                            report.report_file_path = report_file_path
                            await report.complete()
                            logger.info(
                                f"✅ Analysis report saved to: {report_file_path}"
                            )

                            message = f"Answer found from file analysis.\n\nTask: {task}\n\nAnswer: {round_summary.answer}, Report saved to: {report_file_path}"

                            return ToolResponse(
                                success=True,
                                message=message,
                                extra=ToolExtra(
                                    file_path=report_file_path,
                                    data={
                                        "task": task,
                                        "round": round_num,
                                        "answer_found": True,
                                        "answer": round_summary.answer,
                                        "file_path": report_file_path,
                                    },
                                ),
                            )
                        else:
                            message = f"Answer found from file analysis.\n\nTask: {task}\n\nAnswer: {round_summary.answer}"
                            return ToolResponse(success=True, message=message)
                    else:
                        summaries.append(round_summary)

                # 说明相关实现细节。
                round_summary = await self._summarize_summaries(task, round_summaries)
                round_content += (
                    f"### Round {round_num} Summary\n\n{round_summary.summary}\n\n"
                )
                if round_summary.found_answer:
                    round_content += f"**Answer Found**: Yes\n\n**Answer**: {round_summary.answer}\n\n"
                else:
                    round_content += "**Answer Found**: No\n\n"

                await report.add_item(content=round_content)

            # 说明相关实现细节。
            final_summary = await self._summarize_summaries(task, summaries)
            final_content = f"## Final Summary\n\n{final_summary.summary}\n\n"
            if final_summary.found_answer:
                final_content += (
                    f"**Answer Found**: Yes\n\n**Answer**: {final_summary.answer}\n\n"
                )
            else:
                final_content += "**Answer Found**: No\n\n"
            await report.add_item(content=final_content)

            if report_file_path:
                report.report_file_path = report_file_path
                await report.complete()
                logger.info(f"✅ Analysis report saved to: {report_file_path}")

                # 创建所需对象。
                status_text = (
                    "Answer found"
                    if final_summary.found_answer
                    else "No definitive answer found"
                )
                if final_summary.found_answer:
                    answer_text = f"Answer: {final_summary.answer}"
                else:
                    summaries_list = [f"- {s.summary}" for s in summaries[-10:]]
                    answer_text = "Summaries:\n" + "\n".join(summaries_list)

                message = f"Analysis completed after {self.max_rounds} rounds.\n\nTask: {task}\n\n{status_text}.\n\n{answer_text}"
                message += f"\n\nReport saved to: {report_file_path}"

                return ToolResponse(
                    success=final_summary.found_answer,
                    message=message,
                    extra=ToolExtra(
                        file_path=report_file_path,
                        data={
                            "task": task,
                            "rounds": self.max_rounds,
                            "answer_found": final_summary.found_answer,
                            "answer": final_summary.answer
                            if final_summary.found_answer
                            else None,
                            "file_path": report_file_path,
                        },
                    ),
                )
            else:
                if final_summary.found_answer:
                    message = f"Answer found from all file analysis.\n\nTask: {task}\n\nAnswer: {final_summary.answer}"
                    return ToolResponse(success=True, message=message)
                else:
                    message = (
                        f"Analysis completed after {self.max_rounds} rounds but no definitive answer found.\n\nTask: {task}\n\nSummaries:\n"
                        + "\n".join([f"- {s.summary}" for s in summaries[-10:]])
                    )
                    return ToolResponse(success=False, message=message)

        except Exception as e:
            logger.error(f"| ❌ Error in deep analysis: {e}")
            return ToolResponse(
                success=False, message=f"Error during deep analysis: {e!s}"
            )

    async def _get_overall_file_summary(
        self, task: str, files: list[str]
    ) -> str | None:
        """实现 `_get_overall_file_summary` 的业务逻辑。"""
        try:
            # 处理文件与路径。
            file_infos = []
            for file_path in files:
                try:
                    if self._is_url(file_path):
                        # 创建所需对象。
                        file_infos.append(
                            {
                                "path": file_path,
                                "name": file_path,  # 说明相关实现细节。
                                "info": {
                                    "type": "url",
                                    "url": file_path,
                                    "url_type": self._get_url_type(file_path)
                                    or "unknown",
                                },
                            }
                        )
                    else:
                        file_info = get_file_info(file_path)
                        file_infos.append(
                            {
                                "path": file_path,
                                "name": os.path.basename(file_path),
                                "info": file_info,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to get info for {file_path}: {e}")

            if not file_infos:
                return None

            # 转换并规范化数据。
            files_info_text = chr(10).join(
                [
                    dedent(f"""
                File: {info["name"]}
                Path: {info["path"]}
                {
                        ("URL Type: " + info["info"].get("url_type", "unknown"))
                        if info["info"].get("type") == "url"
                        else (
                            f"Size: {info['info'].get('size', 'unknown')}"
                            + chr(10)
                            + f"Created: {info['info'].get('created', 'unknown')}"
                            + chr(10)
                            + f"Modified: {info['info'].get('modified', 'unknown')}"
                        )
                    }
                """).strip()
                    for info in file_infos
                ]
            )

            prompt = dedent(f"""Analyze the following task and provide a summary based on the file information provided.

            Task: {task}

            File Information:
            {files_info_text}

            Based on the file information (sizes, types, names, timestamps, etc.), provide a summary that:
            1. Describes what information can be found from the file metadata
            2. Answers the task if it can be answered from file information alone (e.g., file sizes, video durations, file counts, etc.)
            3. If the task requires file content analysis, indicate what needs to be analyzed

            Provide a concise summary (3-5 sentences).
            """)

            messages = [
                SystemMessage(
                    content="You are an expert at analyzing file metadata and determining if questions can be answered from file information alone."
                ),
                HumanMessage(content=prompt),
            ]

            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=SummaryResponse,
            )

            if not response.success:
                logger.warning(f"Model call failed: {response.message}")
                return None

            if response.extra and response.extra.parsed_model:
                summary_response = response.extra.parsed_model
                summary = Summary(
                    summary=summary_response.summary,
                    found_answer=summary_response.found_answer,
                    answer=summary_response.answer,
                )

                logger.info("| ✅ Overall file summary generated")
                return summary
            else:
                logger.warning(f"Failed to parse response: {response.message}")
                return None

        except Exception as e:
            logger.warning(f"Failed to generate overall file summary: {e}")
            return None

    async def _summarize_summaries(
        self, task: str, summaries: list[Summary]
    ) -> Summary:
        """实现 `_summarize_summaries` 的业务逻辑。"""
        try:
            if not summaries:
                return Summary(
                    summary="No summaries to summarize.",
                    found_answer=False,
                    answer=None,
                )

            # 说明相关实现细节。
            summaries_text = "\n".join([f"- {s.summary}" for s in summaries])

            prompt = dedent(f"""Based on the following analysis summaries, provide a comprehensive summary.

            Task: {task}

            Analysis summaries:
            {summaries_text}

            Synthesize all the information from the summaries and provide:
            1. A comprehensive summary (3-5 sentences) that integrates all findings
            2. Determine if we have found the answer to the task based on all summaries
            3. If the answer is found, provide it in the answer field
            """)

            messages = [
                SystemMessage(
                    content="You are an expert at synthesizing information from multiple analysis summaries."
                ),
                HumanMessage(content=prompt),
            ]

            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=SummaryResponse,
            )

            if not response.success:
                summary_text = (
                    response.message.strip()
                    if response.message
                    else "Model call failed"
                )
                return Summary(summary=summary_text, found_answer=False, answer=None)
            elif response.extra and response.extra.parsed_model:
                summary_response = response.extra.parsed_model
                return Summary(
                    summary=summary_response.summary,
                    found_answer=summary_response.found_answer,
                    answer=summary_response.answer,
                )
            else:
                # 执行回退或重试逻辑。
                summary_text = response.message.strip()
                return Summary(summary=summary_text, found_answer=False, answer=None)

        except Exception as e:
            logger.error(f"| ❌ Error summarizing summaries: {e}")
            return Summary(
                summary=f"Error summarizing summaries: {e}",
                found_answer=False,
                answer=None,
            )

    async def _analyze_task_only(
        self, task: str, summaries: list[Summary], report: Report | None = None
    ) -> None:
        """实现 `_analyze_task_only` 的业务逻辑。"""
        try:
            logger.info("| 🧠 Analyzing task directly (no files)")

            # 执行异步任务。
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"| 🔄 Analysis round {round_num}/{self.max_rounds}")

                prompt = dedent(f"""Analyze the following task step by step. This could be a text game, math problem, logic puzzle, or reasoning task.

                Task: {task}

                For this round, perform detailed analysis:
                1. Break down the task into components
                2. Identify key information and constraints
                3. Apply logical reasoning or mathematical operations
                4. Generate insights and partial solutions
                5. If you find the complete answer, clearly state it

                Provide a concise summary (2-4 sentences) of your analysis for this round.
                """)

                messages = [
                    SystemMessage(
                        content="You are an expert at solving complex reasoning tasks, text games, math problems, and logic puzzles."
                    ),
                    HumanMessage(content=prompt),
                ]

                response = await model_manager(
                    model=self.model_name, messages=messages, response_format=Summary
                )

                if not response.success:
                    summary_text = (
                        response.message.strip()
                        if response.message
                        else "Model call failed"
                    )
                    summary = Summary(
                        summary=summary_text, found_answer=False, answer=None
                    )
                elif response.extra and response.extra.parsed_model:
                    summary = response.extra.parsed_model
                    # 转换并规范化数据。
                else:
                    # 执行回退或重试逻辑。
                    summary_text = response.message.strip()
                    summary = Summary(
                        summary=summary_text, found_answer=False, answer=None
                    )

                summaries.append(summary)

                # 说明相关实现细节。
                if report:
                    round_content = f"## Round {round_num}\n\n{summary.summary}\n\n"
                    if summary.found_answer:
                        round_content += (
                            f"**Answer Found**: Yes\n\n**Answer**: {summary.answer}\n\n"
                        )
                    else:
                        round_content += "**Answer Found**: No\n\n"
                    await report.add_item(round_content)

                # 校验输入与当前状态。
                if summary.found_answer:
                    logger.info(
                        f"| ✅ Answer found in round {round_num}, early stopping."
                    )
                    return

            logger.info(f"| ✅ Task analysis completed after {self.max_rounds} rounds")

        except Exception as e:
            logger.error(f"| ❌ Error analyzing task: {e}")
            error_summary = Summary(
                summary=f"Error analyzing task: {e}", found_answer=False, answer=None
            )
            summaries.append(error_summary)

            # 处理异常情况。
            if report:
                error_content = f"## Error\n\nError analyzing task: {e}\n\n"
                await report.add_item(error_content)

            return

    async def _analyze_text_file(
        self, task: str, file: str, summaries: list[Summary]
    ) -> None:
        """实现 `_analyze_text_file` 的业务逻辑。"""
        try:
            # 加载所需数据。
            local_file_path = file
            if self._is_url(file):
                logger.info("| 📄 Text file URL detected, downloading to local...")
                downloaded_path = await self._download_file(file, file_type="text")
                if not downloaded_path:
                    logger.warning(
                        f"| ❌ Failed to download text file from URL: {file}"
                    )
                    summaries.append(
                        Summary(
                            summary=f"Failed to download text file from URL: {file}",
                            found_answer=False,
                            answer=None,
                        )
                    )
                    return
                local_file_path = downloaded_path
                logger.info(
                    f"| 📄 Using downloaded text file: {os.path.basename(local_file_path)}"
                )

            # 处理文件与路径。
            file_info = get_file_info(local_file_path)
            logger.info(
                f"| 📄 Processing text file: {os.path.basename(local_file_path)} ({file_info.get('size', 'unknown')} bytes)"
            )

            # 加载所需数据。
            _, ext = os.path.splitext(local_file_path.lower())
            if ext == ".md":
                # 加载所需数据。
                logger.info("| 📄 File is already markdown format, using directly")
                if self.base_dir:
                    # 处理文件与路径。
                    base_name = os.path.splitext(os.path.basename(local_file_path))[0]
                    saved_path = os.path.join(self.base_dir, f"{base_name}.md")
                    shutil.copy2(local_file_path, saved_path)
                else:
                    saved_path = local_file_path
            else:
                # 持久化相关数据。
                mdify_response = await self.mdify_tool(
                    file_path=local_file_path, output_format="markdown"
                )
                if mdify_response.extra and mdify_response.extra.file_path:
                    saved_path = mdify_response.extra.file_path
                else:
                    logger.error(
                        f"| ❌ Failed to convert file to markdown: {local_file_path}"
                    )
                    summaries.append(
                        Summary(
                            summary=f"Failed to convert file to markdown: {local_file_path}",
                            found_answer=False,
                            answer=None,
                        )
                    )
                    return

            # 加载所需数据。
            lines = await read_lines_file(saved_path, errors="ignore")

            total_lines = len(lines)
            total_chunks = (total_lines + self.chunk_size - 1) // self.chunk_size

            # 说明相关实现细节。
            for chunk_num in range(1, total_chunks + 1):
                logger.info(
                    f"| 🔄 Analyzing text file chunk {chunk_num}/{total_chunks}"
                )

                # 说明相关实现细节。
                start_line = (chunk_num - 1) * self.chunk_size
                end_line = min(start_line + self.chunk_size, total_lines)
                chunk_lines = lines[start_line:end_line]
                chunk_text = "".join(chunk_lines)

                summary = await self._analyze_markdown_chunk(
                    task, chunk_text, chunk_num, start_line + 1, end_line
                )
                summaries.append(summary)

                if summary.found_answer:
                    logger.info(
                        f"| ✅ Answer found in chunk {chunk_num}, early stopping."
                    )
                    return

            logger.info("| ✅ All chunks of text file analyzed")

        except Exception as e:
            logger.error(f"| ❌ Error analyzing text file {file}: {e}")
            summaries.append(
                Summary(
                    summary=f"Error analyzing text file {file}: {e}",
                    found_answer=False,
                    answer=None,
                )
            )
            return

    async def _analyze_pdf_file(
        self, task: str, file: str, summaries: list[Summary]
    ) -> None:
        """实现 `_analyze_pdf_file` 的业务逻辑。"""
        try:
            # 处理模型调用。
            logger.info("| 📄 Step 1: Trying LLM direct analysis of PDF")

            # 处理模型调用。
            # 说明相关实现细节。
            # 转换并规范化数据。
            if self._is_url(file):
                # 说明相关实现细节。
                pdf_url_value = file
                logger.info(f"| 📄 Using PDF URL: {file}")
            else:
                # 转换并规范化数据。
                pdf_url_value = make_file_url(file_path=file)
                logger.info(f"| 📄 Using local PDF file: {os.path.basename(file)}")

            # 创建所需对象。
            pdf_url = PdfURL(url=pdf_url_value)
            messages = [
                SystemMessage(
                    content="You are an expert at analyzing PDF documents and extracting key information."
                ),
                HumanMessage(
                    content=[
                        ContentPartText(
                            text=dedent(f"""Analyze the following PDF document to answer the task.

                    Task: {task}

                    Extract key information from the PDF that helps answer the task.
                    If the PDF contains the answer to the task, clearly state it.
                    """)
                        ),
                        ContentPartPdf(pdf_url=pdf_url),
                    ]
                ),
            ]

            # 处理模型调用。
            try:
                response = await model_manager(
                    model=self.file_model_name,
                    messages=messages,
                    response_format=SummaryResponse,
                )

                if response.extra and response.extra.parsed_model:
                    summary_response = response.extra.parsed_model
                    summary = Summary(
                        summary=summary_response.summary,
                        found_answer=summary_response.found_answer,
                        answer=summary_response.answer,
                    )
                    summaries.append(summary)

                    if summary.found_answer:
                        logger.info(
                            "| ✅ Answer found via LLM direct analysis, early stopping."
                        )
                        return
                    else:
                        logger.info(
                            "| ⚠️ LLM direct analysis did not find answer, proceeding to chunk-based analysis"
                        )
                else:
                    logger.warning(
                        "| ⚠️ LLM direct analysis failed to parse response, proceeding to chunk-based analysis"
                    )
            except Exception as e:
                logger.warning(
                    f"| ⚠️ LLM direct analysis failed: {e}, proceeding to chunk-based analysis"
                )

            # 转换并规范化数据。
            logger.info(
                "| 📄 Step 2: Converting PDF to markdown and analyzing in chunks"
            )

            # 校验输入与当前状态。
            if self._is_url(file):
                logger.info(f"| 📄 Processing PDF URL: {file}")
                # 检索所需信息。
                doc_result = await fetch_url(file)
                if not doc_result or not doc_result.markdown:
                    logger.warning(f"Failed to fetch PDF from URL: {file}")
                    summaries.append(
                        Summary(
                            summary=f"Failed to fetch PDF from URL: {file}",
                            found_answer=False,
                            answer=None,
                        )
                    )
                    return

                # 持久化相关数据。
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False, dir=self.base_dir
                ) as tmp_file:
                    tmp_file.write(doc_result.markdown)
                    saved_path = tmp_file.name
            else:
                # 处理文件与路径。
                file_info = get_file_info(file)
                logger.info(
                    f"| 📄 Processing PDF file: {os.path.basename(file)} ({file_info.get('size', 'unknown')} bytes)"
                )

                # 持久化相关数据。
                mdify_response = await self.mdify_tool(
                    file_path=file, output_format="markdown"
                )
                if mdify_response.extra and mdify_response.extra.file_path:
                    saved_path = mdify_response.extra.file_path

            # 加载所需数据。
            lines = await read_lines_file(saved_path, errors="ignore")

            total_lines = len(lines)
            total_chunks = (total_lines + self.chunk_size - 1) // self.chunk_size

            # 说明相关实现细节。
            for chunk_num in range(1, total_chunks + 1):
                logger.info(f"| 🔄 Analyzing PDF file chunk {chunk_num}/{total_chunks}")

                # 说明相关实现细节。
                start_line = (chunk_num - 1) * self.chunk_size
                end_line = min(start_line + self.chunk_size, total_lines)
                chunk_lines = lines[start_line:end_line]
                chunk_text = "".join(chunk_lines)

                summary = await self._analyze_markdown_chunk(
                    task, chunk_text, chunk_num, start_line + 1, end_line
                )
                summaries.append(summary)

                if summary.found_answer:
                    logger.info(
                        f"| ✅ Answer found in chunk {chunk_num}, early stopping."
                    )
                    return

            logger.info("| ✅ All chunks of PDF file analyzed")

        except Exception as e:
            logger.error(f"| ❌ Error analyzing PDF file {file}: {e}")
            summaries.append(
                Summary(
                    summary=f"Error analyzing PDF file {file}: {e}",
                    found_answer=False,
                    answer=None,
                )
            )
            return

    async def _analyze_markdown_chunk(
        self, task: str, chunk_text: str, chunk_num: int, start_line: int, end_line: int
    ) -> Summary:
        """实现 `_analyze_markdown_chunk` 的业务逻辑。"""
        try:
            logger.info(
                f"| 🔍 Analyzing chunk {chunk_num} (lines {start_line}-{end_line})"
            )

            context = f"Task: {task}\n\n"
            context += f"Current chunk (lines {start_line}-{end_line}):\n{chunk_text}"

            prompt = dedent(f"""Analyze this chunk of the document and extract information relevant to the task.

            {context}

            Extract key information that helps answer the task. Provide a concise summary (2-3 sentences) of findings from this chunk.
            If this chunk contains the answer to the task, set found_answer to True and provide the answer in the answer field.
            """)

            messages = [
                SystemMessage(
                    content="You are an expert at extracting key information from documents."
                ),
                HumanMessage(content=prompt),
            ]

            response = await model_manager(
                model=self.model_name, messages=messages, response_format=Summary
            )

            if not response.success:
                summary_text = (
                    response.message.strip()
                    if response.message
                    else "Model call failed"
                )
                return Summary(summary=summary_text, found_answer=False, answer=None)
            elif response.extra and response.extra.parsed_model:
                parsed_summary = response.extra.parsed_model
                # 转换并规范化数据。
                return parsed_summary
            else:
                summary = response.message.strip()
                return Summary(summary=summary, found_answer=False, answer=None)

        except Exception as e:
            logger.error(f"| ❌ Error analyzing markdown chunk: {e}")
            return Summary(
                summary=f"Error analyzing markdown chunk: {e}",
                found_answer=False,
                answer=None,
            )

    async def _analyze_image_file(
        self, task: str, file: str, summaries: list[Summary]
    ) -> None:
        """实现 `_analyze_image_file` 的业务逻辑。"""
        try:
            # 校验输入与当前状态。
            is_url = self._is_url(file)
            local_file_path = file

            # 加载所需数据。
            if is_url:
                logger.info(f"| 📥 Downloading image from URL: {file}")
                local_file_path = await self._download_file(file, "image")
                if local_file_path is None:
                    logger.error(f"| ❌ Failed to download image from URL: {file}")
                    return
                logger.info(f"| ✅ Image downloaded successfully: {local_file_path}")
            elif not os.path.exists(file):
                logger.warning(f"Image file not found: {file}")
                return

            # 处理模型调用。
            logger.info("| 🖼️ Step 1: Trying LLM direct analysis of image")

            # 处理模型调用。
            # 转换并规范化数据。
            # 加载所需数据。
            image_url_value = make_file_url(file_path=local_file_path)
            logger.info(f"| 🖼️ Using image file: {os.path.basename(local_file_path)}")

            # 创建所需对象。
            image_url = ImageURL(url=image_url_value, detail="high")
            messages = [
                SystemMessage(
                    content="You are an expert at analyzing images and extracting visual information."
                ),
                HumanMessage(
                    content=[
                        ContentPartText(
                            text=dedent(f"""Analyze the following image to answer the task.

                    Task: {task}

                    Extract key information from the image that helps answer the task.
                    Focus on visual elements, text in images, patterns, and any relevant details.
                    If the image contains the answer to the task, clearly state it.
                    """)
                        ),
                        ContentPartImage(image_url=image_url),
                    ]
                ),
            ]

            # 处理模型调用。
            try:
                response = await model_manager(
                    model=self.model_name,
                    messages=messages,
                    response_format=SummaryResponse,
                )

                if not response.success:
                    logger.warning(
                        f"| ⚠️ LLM direct analysis failed: {response.message}, proceeding to multi-step analysis"
                    )
                elif response.extra and response.extra.parsed_model:
                    summary_response = response.extra.parsed_model
                    summary = Summary(
                        summary=summary_response.summary,
                        found_answer=summary_response.found_answer,
                        answer=summary_response.answer,
                    )
                    summaries.append(summary)

                    if summary.found_answer:
                        logger.info(
                            "| ✅ Answer found via LLM direct analysis, early stopping."
                        )
                        return
                    else:
                        logger.info(
                            "| ⚠️ LLM direct analysis did not find answer, proceeding to multi-step analysis"
                        )
                else:
                    logger.warning(
                        "| ⚠️ LLM direct analysis failed to parse response, proceeding to multi-step analysis"
                    )
            except Exception as e:
                logger.warning(
                    f"| ⚠️ LLM direct analysis failed: {e}, proceeding to multi-step analysis"
                )

            # 说明相关实现细节。
            logger.info("| 🖼️ Step 2: Analyzing image with multiple steps")

            for step_num in range(1, self.max_steps + 1):
                logger.info(f"| 🔄 Analyzing image step {step_num}/{self.max_steps}")

                # 创建所需对象。
                image_url = ImageURL(url=image_url_value, detail="high")
                messages = [
                    SystemMessage(
                        content="You are an expert at analyzing images and extracting visual information."
                    ),
                    HumanMessage(
                        content=[
                            ContentPartText(
                                text=dedent(f"""Analyze the following image to answer the task.

                        Task: {task}

                        Extract key information from the image that helps answer the task.
                        Focus on visual elements, text in images, patterns, and any relevant details.
                        """)
                            ),
                            ContentPartImage(image_url=image_url),
                        ]
                    ),
                ]

                response = await model_manager(
                    model=self.model_name, messages=messages, response_format=Summary
                )

                if not response.success:
                    summary_text = (
                        response.message.strip()
                        if response.message
                        else "Model call failed"
                    )
                    summary = Summary(
                        summary=summary_text, found_answer=False, answer=None
                    )
                elif response.extra and response.extra.parsed_model:
                    summary = response.extra.parsed_model
                    # 转换并规范化数据。
                else:
                    # 执行回退或重试逻辑。
                    summary_text = response.message.strip()
                    summary = Summary(
                        summary=summary_text, found_answer=False, answer=None
                    )

                summaries.append(summary)

                # 校验输入与当前状态。
                if summary.found_answer:
                    logger.info(
                        f"| ✅ Answer found in image step {step_num}, early stopping."
                    )
                    return

            logger.info(f"| ✅ Image analysis completed after {self.max_steps} steps")

        except Exception as e:
            logger.error(f"| ❌ Error analyzing image file {file}: {e}")
            summaries.append(
                Summary(
                    summary=f"Error analyzing image file {file}: {e}",
                    found_answer=False,
                    answer=None,
                )
            )
            return

    async def _analyze_audio_file(
        self, task: str, file: str, summaries: list[Summary]
    ) -> None:
        """实现 `_analyze_audio_file` 的业务逻辑。"""
        try:
            # 加载所需数据。
            local_file_path = file
            if self._is_url(file):
                logger.info("| 🎵 Audio URL detected, downloading to local...")
                downloaded_path = await self._download_file(file, file_type="audio")
                if not downloaded_path:
                    logger.warning(f"| ❌ Failed to download audio from URL: {file}")
                    summaries.append(
                        Summary(
                            summary=f"Failed to download audio from URL: {file}",
                            found_answer=False,
                            answer=None,
                        )
                    )
                    return
                local_file_path = downloaded_path
                logger.info(
                    f"| 🎵 Using downloaded audio file: {os.path.basename(local_file_path)}"
                )

            if not os.path.exists(local_file_path):
                logger.warning(f"Audio file not found: {local_file_path}")
                return

            # 处理文件与路径。
            file_info = get_file_info(local_file_path)
            logger.info(
                f"| 🎵 Processing audio file: {os.path.basename(local_file_path)} ({file_info.get('size', 'unknown')} bytes)"
            )

            # 转换并规范化数据。
            audio_url_value = make_file_url(file_path=local_file_path)

            # 创建所需对象。
            audio_url = AudioURL(url=audio_url_value)
            messages = [
                SystemMessage(
                    content="You are an expert at analyzing audio files, transcribing speech, and extracting key information."
                ),
                HumanMessage(
                    content=[
                        ContentPartText(
                            text=dedent(f"""Analyze the following audio file to answer the task.

                    Task: {task}

                    Transcribe the audio and extract key information that helps answer the task.
                    If the audio contains the answer to the task, clearly state it.
                    """)
                        ),
                        ContentPartAudio(audio_url=audio_url),
                    ]
                ),
            ]

            # 处理模型调用。
            try:
                response = await model_manager(
                    model=self.file_model_name,
                    messages=messages,
                    response_format=SummaryResponse,
                )

                if not response.success:
                    logger.warning(
                        f"| ⚠️ LLM direct analysis failed: {response.message}"
                    )
                elif response.extra and response.extra.parsed_model:
                    summary_response = response.extra.parsed_model
                    summary = Summary(
                        summary=summary_response.summary,
                        found_answer=summary_response.found_answer,
                        answer=summary_response.answer,
                    )
                    summaries.append(summary)

                    if summary.found_answer:
                        logger.info("| ✅ Answer found via LLM direct analysis.")
                        return
                    else:
                        logger.info("| ⚠️ LLM direct analysis did not find answer.")
                else:
                    logger.warning("| ⚠️ LLM direct analysis failed to parse response.")
            except Exception as e:
                logger.error(f"| ❌ Error in LLM audio analysis: {e}")
                summaries.append(
                    Summary(
                        summary=f"Error analyzing audio file {local_file_path}: {e}",
                        found_answer=False,
                        answer=None,
                    )
                )
                return

        except Exception as e:
            logger.error(f"| ❌ Error analyzing audio file {file}: {e}")
            summaries.append(
                Summary(
                    summary=f"Error analyzing audio file {file}: {e}",
                    found_answer=False,
                    answer=None,
                )
            )
            return

    async def _analyze_video_file(
        self, task: str, file: str, summaries: list[Summary]
    ) -> None:
        """实现 `_analyze_video_file` 的业务逻辑。"""
        try:
            # 校验输入与当前状态。
            is_url = self._is_url(file)
            local_file_path = file

            if is_url:
                if self._is_youtube_url(file):
                    # 处理模型调用。
                    video_url_value = file
                    logger.info(f"| 🎬 Using YouTube video URL: {file}")
                else:
                    # 加载所需数据。
                    logger.info(
                        "| 🎬 Non-YouTube video URL detected, downloading to local..."
                    )
                    downloaded_path = await self._download_file(file, file_type="video")
                    if not downloaded_path:
                        logger.warning(
                            f"| ❌ Failed to download video from URL: {file}"
                        )
                        summaries.append(
                            Summary(
                                summary=f"Failed to download video from URL: {file}",
                                found_answer=False,
                                answer=None,
                            )
                        )
                        return
                    local_file_path = downloaded_path
                    # 转换并规范化数据。
                    video_url_value = make_file_url(file_path=local_file_path)
                    logger.info(
                        f"| 🎬 Using downloaded video file: {os.path.basename(local_file_path)}"
                    )
            else:
                # 校验输入与当前状态。
                if not os.path.exists(file):
                    logger.warning(f"Video file not found: {file}")
                    return

                # 转换并规范化数据。
                video_url_value = make_file_url(file_path=file)
                logger.info(f"| 🎬 Using local video file: {os.path.basename(file)}")

            # 处理模型调用。
            logger.info("| 🎬 Step 1: Trying LLM direct analysis of video")

            # 创建所需对象。
            video_url = VideoURL(url=video_url_value)
            messages = [
                SystemMessage(
                    content="You are an expert at analyzing videos and extracting key information."
                ),
                HumanMessage(
                    content=[
                        ContentPartText(
                            text=dedent(f"""Analyze the following video to answer the task.

                    Task: {task}

                    Extract key information from the video that helps answer the task.
                    If the video contains the answer to the task, clearly state it.
                    """)
                        ),
                        ContentPartVideo(video_url=video_url),
                    ]
                ),
            ]

            # 处理模型调用。
            try:
                response = await model_manager(
                    model=self.file_model_name,
                    messages=messages,
                    response_format=SummaryResponse,
                )

                if response.extra and response.extra.parsed_model:
                    summary_response = response.extra.parsed_model
                    summary = Summary(
                        summary=summary_response.summary,
                        found_answer=summary_response.found_answer,
                        answer=summary_response.answer,
                    )
                    summaries.append(summary)

                    if summary.found_answer:
                        logger.info(
                            "| ✅ Answer found via LLM direct analysis, early stopping."
                        )
                        return
                    else:
                        logger.info(
                            "| ⚠️ LLM direct analysis did not find answer, proceeding to chunk-based analysis"
                        )
                else:
                    logger.warning(
                        "| ⚠️ LLM direct analysis failed to parse response, proceeding to chunk-based analysis"
                    )
            except Exception as e:
                logger.warning(
                    f"| ⚠️ LLM direct analysis failed: {e}, proceeding to chunk-based analysis"
                )

            # 转换并规范化数据。
            logger.info(
                "| 🎬 Step 2: Converting video to markdown and analyzing in chunks"
            )

            # 加载所需数据。
            if is_url and self._is_youtube_url(file):
                # 检索所需信息。
                logger.info(f"| 🎬 Processing YouTube video URL: {file}")
                doc_result = await fetch_url(file)
                if not doc_result or not doc_result.markdown:
                    logger.warning(f"Failed to fetch video content from URL: {file}")
                    summaries.append(
                        Summary(
                            summary=f"Failed to fetch video content from URL: {file}",
                            found_answer=False,
                            answer=None,
                        )
                    )
                    return

                # 持久化相关数据。
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False, dir=self.base_dir
                ) as tmp_file:
                    tmp_file.write(doc_result.markdown)
                    saved_path = tmp_file.name
            else:
                # 加载所需数据。
                file_info = get_file_info(local_file_path)
                logger.info(
                    f"| 🎬 Processing video file: {os.path.basename(local_file_path)} ({file_info.get('size', 'unknown')} bytes)"
                )

                # 持久化相关数据。
                mdify_response = await self.mdify_tool(
                    file_path=local_file_path, output_format="markdown"
                )
                # 持久化相关数据。
                if mdify_response.extra and mdify_response.extra.file_path:
                    saved_path = mdify_response.extra.file_path
                elif mdify_response.extra and mdify_response.extra.data:
                    saved_path = mdify_response.extra.data.get("saved_path")
                else:
                    raise ValueError("mdify_tool did not return saved_path in extra")

            # 加载所需数据。
            lines = await read_lines_file(saved_path, errors="ignore")

            total_lines = len(lines)
            total_chunks = (total_lines + self.chunk_size - 1) // self.chunk_size

            # 说明相关实现细节。
            for chunk_num in range(1, total_chunks + 1):
                logger.info(
                    f"| 🔄 Analyzing video file chunk {chunk_num}/{total_chunks}"
                )

                # 说明相关实现细节。
                start_line = (chunk_num - 1) * self.chunk_size
                end_line = min(start_line + self.chunk_size, total_lines)
                chunk_lines = lines[start_line:end_line]
                chunk_text = "".join(chunk_lines)

                summary = await self._analyze_markdown_chunk(
                    task, chunk_text, chunk_num, start_line + 1, end_line
                )
                summaries.append(summary)

                if summary.found_answer:
                    logger.info(
                        f"| ✅ Answer found in chunk {chunk_num}, early stopping."
                    )
                    return

            logger.info("| ✅ All chunks of video file analyzed")

        except Exception as e:
            logger.error(f"| ❌ Error analyzing video file {file}: {e}")
            summaries.append(
                Summary(
                    summary=f"Error analyzing video file {file}: {e}",
                    found_answer=False,
                    answer=None,
                )
            )
            return

    async def _validate_file(self, file_path: str) -> bool:
        """实现 `_validate_file` 的业务逻辑。"""
        try:
            # 校验输入与当前状态。
            if self._is_url(file_path):
                url_type = self._get_url_type(file_path)
                if url_type:
                    # 校验输入与当前状态。
                    if url_type == "video" and not self._is_youtube_url(file_path):
                        logger.warning(f"Video URL must be YouTube: {file_path}")
                        return False
                    return True
                else:
                    logger.warning(f"Unsupported URL type: {file_path}")
                    return False

            # 校验输入与当前状态。
            if not os.path.exists(file_path):
                logger.warning(f"File does not exist: {file_path}")
                return False

            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                logger.warning(f"File too large: {file_path} ({file_size} bytes)")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return False
