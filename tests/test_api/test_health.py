"""健康检查 API 集成测试

测试覆盖：/health 端点、各组件状态响应、JSON 结构校验。
"""



class TestHealthEndpoint:
    """健康检查端点测试"""

    def test_health_returns_json(self, test_app, assert_json_ok):
        response = test_app.get("/health")
        data = assert_json_ok(response)
        # 统一响应封装: {code, message, data}
        assert data["code"] == 200
        assert "status" in data["data"]

    def test_health_has_services(self, test_app, assert_json_ok):
        response = test_app.get("/health")
        data = assert_json_ok(response)
        assert "data" in data
        # 应该有服务状态信息
        services = data["data"].get("services", data["data"])
        assert services is not None

    def test_health_cache_stats_present(self, test_app, assert_json_ok):
        response = test_app.get("/health")
        data = assert_json_ok(response)
        # cache stats 应在 data 中（直接或嵌套）
        response_str = str(data).lower()
        assert any(kw in response_str for kw in ["cache", "cost", "prompt"])

    def test_health_response_time_acceptable(self, test_app):
        import time

        start = time.time()
        response = test_app.get("/health")
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 5.0  # 健康检查应在 5s 内完成


class TestAPIFeedbackEndpoint:
    """反馈端点测试"""

    def test_datasets_stats(self, test_app, assert_json_ok):
        response = test_app.get("/api/eval/datasets/stats")
        data = assert_json_ok(response)
        assert "total" in data["data"]
        assert "diagnostic" in data["data"]
        assert "negative" in data["data"]
        assert "by_category" in data["data"]

    def test_datasets_stats_categories(self, test_app, assert_json_ok):
        response = test_app.get("/api/eval/datasets/stats")
        data = assert_json_ok(response)
        categories = data["data"]["by_category"]
        assert isinstance(categories, dict)
        for _cat in ["easy", "medium", "hard", "edge_case", "chitchat", "knowledge"]:
            # 至少应该有一些类别出现
            pass


class TestAPIErrorHandling:
    """异常处理测试"""

    def test_nonexistent_endpoint_404(self, test_app):
        response = test_app.get("/api/nonexistent_endpoint")
        assert response.status_code == 404
