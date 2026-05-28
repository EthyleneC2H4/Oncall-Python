"""输入安全检测中间件

双层防护：
1. 规则层：检测常见 Prompt 注入模式
2. PII 检测：手机号、身份证号、邮箱、银行卡号
"""

import re
from typing import Optional

from loguru import logger


class InputGuard:
    """输入安全检测器"""

    # Prompt 注入检测规则（中英文）
    INJECTION_PATTERNS = [
        r"忽略.*(?:前面|之前|上面).*(?:指令|提示|内容)",
        r"你现在是",
        r"你的新角色是",
        r"(?:忘记|忽略|无视).*(?:系统|角色|指令)",
        r"system\s*:",
        r"<\s*system\s*>",
        r"ignore\s+(?:previous|above|all)\s+instructions?",
        r"you\s+are\s+now",
        r"new\s+(?:role|persona|identity)",
        r"forget\s+(?:everything|all|your)",
        r"pretend\s+(?:you|to\s+be)",
        r"jailbreak",
        r"DAN\s+mode",
        r"\[SYSTEM\]",
        r"OVERRIDE",
    ]

    # PII 检测正则
    PII_PATTERNS = {
        "phone": r"1[3-9]\d{9}",
        "id_card": r"\d{17}[\dXx]",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "bank_card": r"\d{16,19}",
    }

    def __init__(self):
        self._injection_re = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        self._pii_re = {k: re.compile(p) for k, p in self.PII_PATTERNS.items()}

    def check_injection(self, text: str) -> Optional[str]:
        """检测 Prompt 注入

        Returns:
            检测到的注入模式描述，None 表示安全
        """
        for pattern in self._injection_re:
            match = pattern.search(text)
            if match:
                matched_text = match.group()
                logger.warning(f"检测到可疑 Prompt 注入: '{matched_text}'")
                return f"检测到可疑输入模式: {matched_text}"
        return None

    def detect_pii(self, text: str) -> dict[str, list[str]]:
        """检测 PII 信息

        Returns:
            {pii_type: [matched_values...]}
        """
        found: dict[str, list[str]] = {}
        for pii_type, pattern in self._pii_re.items():
            matches = pattern.findall(text)
            if matches:
                found[pii_type] = matches
        return found

    def mask_pii(self, text: str) -> str:
        """脱敏 PII 信息"""
        result = text
        for pii_type, pattern in self._pii_re.items():
            if pii_type == "phone":
                result = re.sub(pattern, lambda m: m.group()[:3] + "****" + m.group()[-4:], result)
            elif pii_type == "id_card":
                result = re.sub(pattern, lambda m: m.group()[:6] + "********" + m.group()[-4:], result)
            elif pii_type == "email":
                result = re.sub(
                    self.PII_PATTERNS["email"],
                    lambda m: m.group().split("@")[0][:2] + "***@" + m.group().split("@")[1],
                    result,
                )
            elif pii_type == "bank_card":
                result = re.sub(pattern, lambda m: m.group()[:4] + " **** **** " + m.group()[-4:], result)
        return result

    def validate(self, text: str) -> tuple[bool, str, str]:
        """完整输入验证

        Returns:
            (is_safe, message, sanitized_text)
        """
        if not text or not text.strip():
            return True, "", text

        # 检测注入
        injection = self.check_injection(text)
        if injection:
            from app.core.audit import audit_logger
            audit_logger.log_request(
                request_id="input_guard",
                intent="BLOCKED",
                error=injection,
                extra={"blocked_input": text[:200]},
            )
            return False, injection, text

        # 检测并脱敏 PII
        pii_found = self.detect_pii(text)
        sanitized = text
        if pii_found:
            logger.info(f"检测到 PII 信息: {list(pii_found.keys())}，已脱敏")
            sanitized = self.mask_pii(text)

        return True, "", sanitized


# 全局单例
input_guard = InputGuard()
