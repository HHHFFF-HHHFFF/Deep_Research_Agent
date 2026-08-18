from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.logger import logger


class VersionStatus(str, Enum):
    """定义 `VersionStatus`，封装相关数据与行为。"""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class VersionInfo(BaseModel):
    """定义 `VersionInfo`，封装相关数据与行为。"""

    version: str = Field(description="Version string (e.g., '1.0.0', '2.1.3')")
    status: VersionStatus = Field(
        default=VersionStatus.ACTIVE, description="Version status"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )
    description: str | None = Field(default=None, description="Version description")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Version metadata"
    )


class ComponentVersionHistory(BaseModel):
    """定义 `ComponentVersionHistory`，封装相关数据与行为。"""

    name: str = Field(description="Name of the component")
    component_type: str = Field(
        description="Type of component (tool, environment, agent)"
    )
    current_version: str = Field(description="Current active version")
    versions: dict[str, VersionInfo] = Field(
        default_factory=dict, description="Version history records"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Component metadata"
    )

    def add_version(
        self,
        version: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VersionInfo:
        """添加与 `add_version` 对应的数据或状态。"""
        if version in self.versions:
            logger.warning(
                f"| ⚠️ Version {version} already exists for {self.name}, updating..."
            )
            version_info = self.versions[version]
            version_info.updated_at = datetime.now(timezone.utc)
            if description:
                version_info.description = description
            if metadata:
                version_info.metadata.update(metadata)
        else:
            version_info = VersionInfo(
                version=version, description=description, metadata=metadata or {}
            )
            self.versions[version] = version_info

        self.current_version = version

        logger.debug(f"| ✅ Added version record {version} for {self.name}")
        return version_info

    def list_versions(self) -> list[str]:
        """列出与 `list_versions` 对应的数据或状态。"""
        return list(self.versions.keys())

    def deprecate_version(self, version: str):
        """实现 `deprecate_version` 的业务逻辑。"""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found for {self.name}")

        if version == self.current_version:
            raise ValueError(f"Cannot deprecate current version {version}")

        self.versions[version].status = VersionStatus.DEPRECATED
        logger.info(f"| 📝 Deprecated version {version} for {self.name}")

    def archive_version(self, version: str):
        """实现 `archive_version` 的业务逻辑。"""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found for {self.name}")

        self.versions[version].status = VersionStatus.ARCHIVED
        logger.info(f"| 📦 Archived version {version} for {self.name}")
