"""Circuit Breaker 熔断器

为每个外部服务提供独立的熔断保护，防止级联故障：
- CLOSED: 正常运行，追踪失败次数
- OPEN: 服务不可用，所有调用快速失败
- HALF_OPEN: 冷却期过后放行探测请求

用法:
    breaker = CircuitBreaker("llm")
    try:
        breaker.before_call()
        result = await some_api_call()
        breaker.record_success()
    except CircuitOpenError:
        # 走降级路径
    except Exception:
        breaker.record_failure()
        raise
"""

import threading
import time
from enum import StrEnum

from loguru import logger


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """熔断器处于 OPEN 状态时抛出，调用方应走降级路径"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        super().__init__(f"Circuit breaker for '{service_name}' is OPEN")


class CircuitBreaker:
    """单个服务的熔断器"""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # 检查是否应该转换为 HALF_OPEN
                if time.time() - self._last_failure_time >= self.cooldown_seconds:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(
                        f"CircuitBreaker[{self.name}]: OPEN → HALF_OPEN "
                        f"(冷却 {self.cooldown_seconds}s 已过)"
                    )
            return self._state

    @property
    def failure_count(self) -> int:
        """当前连续失败次数（只读观测，供健康检查/监控使用）"""
        with self._lock:
            return self._failure_count

    def before_call(self) -> None:
        """调用前检查，OPEN 状态直接抛出异常"""
        current = self.state  # 触发 OPEN→HALF_OPEN 转换检查
        if current == CircuitState.OPEN:
            raise CircuitOpenError(self.name)

    def record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(f"CircuitBreaker[{self.name}]: HALF_OPEN → CLOSED (探测成功)")
            self._state = CircuitState.CLOSED
            self._failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"CircuitBreaker[{self.name}]: HALF_OPEN → OPEN (探测失败)")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"CircuitBreaker[{self.name}]: CLOSED → OPEN "
                    f"(连续失败 {self._failure_count} 次)"
                )

    def reset(self) -> None:
        """手动重置"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0


# ---------- 全局熔断器注册表 ----------

_breakers: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(name: str) -> CircuitBreaker:
    """获取或创建指定服务的熔断器（阈值/冷却从配置读取）"""
    with _registry_lock:
        if name not in _breakers:
            from app.config import config  # 局部导入避免 import 环

            _breakers[name] = CircuitBreaker(
                name,
                failure_threshold=config.circuit_failure_threshold,
                cooldown_seconds=config.circuit_cooldown_seconds,
            )
        return _breakers[name]


# 预注册核心服务熔断器
BREAKER_LLM = "llm"
BREAKER_EMBEDDING = "embedding"
BREAKER_RERANK = "rerank"
BREAKER_MILVUS = "milvus"
BREAKER_MCP_CLS = "mcp_cls"
BREAKER_MCP_MONITOR = "mcp_monitor"

for _name in [
    BREAKER_LLM,
    BREAKER_EMBEDDING,
    BREAKER_RERANK,
    BREAKER_MILVUS,
    BREAKER_MCP_CLS,
    BREAKER_MCP_MONITOR,
]:
    get_breaker(_name)
