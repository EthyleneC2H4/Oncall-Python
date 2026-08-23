"""OnCall API 压力测试 — Locustfile

测试场景:
1. 健康检查 (低压力/高频): 模拟监控系统定期探测
2. 快速对话 (中等压力): 模拟用户日常问答
3. AIOps 诊断 (高压力/低频): 模拟运维人员触发诊断
4. 反馈提交 (低压力): 模拟诊断后的反馈收集

使用方式:
    # 启动目标服务后:
    pip install locust
    cd tests/load
    locust -f locustfile.py --host=http://localhost:9900

    # 无 UI 模式（CI）:
    locust -f locustfile.py --host=http://localhost:9900 --headless \
        --users 50 --spawn-rate 5 --run-time 120s \
        --html=report.html --csv=report
"""

import random

from locust import HttpUser, between, events, task

# ──────────────── 测试数据 ────────────────

DIAGNOSTIC_QUERIES = [
    "CPU持续高于90%，伴随OOM日志，帮我分析",
    "磁盘使用率超过90%，日志写入变慢",
    "服务响应时间P99超过5秒",
    "服务不可用，健康检查持续失败",
    "内存使用率飙升至95%，GC频繁",
    "数据库连接池耗尽，新请求全部失败",
    "网络延迟突然升高，丢包率增加",
    "应用启动失败，端口被占用",
    "日志中出现大量NullPointerException",
    "线上出现OOM Killer把进程杀了",
]

KNOWLEDGE_QUERIES = [
    "什么是向量数据库",
    "如何配置Milvus索引参数",
    "BM25算法的原理是什么",
    "怎么部署Docker容器",
    "Kubernetes和Docker Compose的区别",
]

CHITCHAT_QUERIES = [
    "你好，你是谁",
    "今天天气怎么样",
    "帮我写一首诗",
]

# 分场景比例: 60% 诊断, 25% 知识, 5% 闲聊, 10% 其他
QUERY_WEIGHTS = [(DIAGNOSTIC_QUERIES, 60), (KNOWLEDGE_QUERIES, 25), (CHITCHAT_QUERIES, 5)]
ALL_QUERIES = DIAGNOSTIC_QUERIES + KNOWLEDGE_QUERIES + CHITCHAT_QUERIES


def random_query() -> str:
    """按权重随机选择查询"""
    r = random.randint(1, 100)
    cumulative = 0
    for pool, weight in QUERY_WEIGHTS:
        cumulative += weight
        if r <= cumulative:
            return random.choice(pool)
    return random.choice(DIAGNOSTIC_QUERIES)


# ──────────────── 用户行为类 ────────────────


class OnCallUser(HttpUser):
    """模拟运维工程师使用 OnCall 的行为模式"""

    wait_time = between(1, 5)  # 操作间隔 1-5 秒

    def on_start(self):
        """用户登录/初始化"""
        self.session_id = f"load-test-{random.randint(10000, 99999)}"

    @task(20)
    def health_check(self):
        """健康检查 — 高频低负载"""
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(15)
    def ask_diagnostic_query(self):
        """诊断查询 — 主要业务场景"""
        query = random.choice(DIAGNOSTIC_QUERIES)
        payload = {
            "query": query,
            "session_id": self.session_id,
        }
        with self.client.post(
            "/api/chat",
            json=payload,
            catch_response=True,
            timeout=30,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == 200:
                    resp.success()
                else:
                    resp.failure(f"API error: {data}")
            elif resp.status_code == 429:
                resp.success()  # 限流是预期行为
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(8)
    def ask_knowledge_query(self):
        """知识查询 — 快速场景"""
        query = random.choice(KNOWLEDGE_QUERIES)
        payload = {
            "query": query,
            "session_id": self.session_id,
        }
        with self.client.post(
            "/api/chat",
            json=payload,
            catch_response=True,
            timeout=15,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 429:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(3)
    def aiops_diagnose(self):
        """AIOps 诊断 — 低频重负载"""
        payload = {
            "task": random.choice(DIAGNOSTIC_QUERIES),
            "session_id": self.session_id,
        }
        with self.client.post(
            "/api/aiops",
            json=payload,
            catch_response=True,
            timeout=60,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 429:
                resp.success()
            else:
                resp.failure(f"AIOps diagnose failed: {resp.status_code}")

    @task(2)
    def submit_feedback(self):
        """提交反馈 — 低频"""
        payload = {
            "session_id": self.session_id,
            "message_index": 0,
            "feedback_type": random.choice(["positive", "negative", "comment"]),
            "comment": "自动化压力测试反馈",
            "actual_root_cause": "MemoryLeak" if random.random() < 0.3 else "",
        }
        with self.client.post(
            "/api/feedback",
            json=payload,
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 422):
                resp.success()
            else:
                resp.failure(f"Feedback failed: {resp.status_code}")

    @task(5)
    def get_dataset_stats(self):
        """评测数据集统计 — 轻量读取"""
        with self.client.get("/api/eval/datasets/stats", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Stats failed: {resp.status_code}")


class AIOpsHeavyUser(HttpUser):
    """模拟重度 AIOps 诊断用户 — 仅 AIOps 场景"""

    wait_time = between(5, 15)

    def on_start(self):
        self.session_id = f"load-aiops-{random.randint(10000, 99999)}"

    @task
    def aiops_diagnose(self):
        query = random.choice(DIAGNOSTIC_QUERIES)
        payload = {
            "task": query,
            "session_id": self.session_id,
        }
        with self.client.post(
            "/api/aiops",
            json=payload,
            catch_response=True,
            timeout=60,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 429:
                resp.success()
            else:
                resp.failure(f"AIOps failed: {resp.status_code}")


# ──────────────── 自定义事件 ────────────────


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """测试开始时的日志"""
    print(f"\n{'='*60}")
    print("OnCall API 压力测试开始")
    print(f"{'='*60}")
    print(f"目标: {environment.host}")
    print("场景: 健康检查 + 对话 + AIOps 诊断 + 反馈")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """测试结束时的汇总"""
    print(f"\n{'='*60}")
    print("压力测试结束")
    print(f"{'='*60}")
    stats = environment.stats
    print(f"总请求数: {stats.total.num_requests}")
    print(f"失败数: {stats.total.num_failures}")
    print(f"平均响应时间: {stats.total.avg_response_time:.0f}ms")
    print(f"P50: {stats.total.get_response_time_percentile(0.5):.0f}ms")
    print(f"P95: {stats.total.get_response_time_percentile(0.95):.0f}ms")
    print(f"P99: {stats.total.get_response_time_percentile(0.99):.0f}ms")
    print(f"RPS: {stats.total.total_rps:.2f}")
    print(f"失败率: {stats.total.fail_ratio:.2%}")


# ──────────────── 分布式运行提示 ────────────────
# 主节点:
#   locust -f locustfile.py --host=http://localhost:9900 --master
# 工作节点:
#   locust -f locustfile.py --host=http://localhost:9900 --worker --master-host=<master_ip>
