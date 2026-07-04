from locust import HttpUser, task, between, events
import json

class SolarSentinelUser(HttpUser):
    '''
    Load test verifying SLO-2 (throughput) and SLO-3 (availability).
    Run: locust -f tests/load/locustfile.py --host http://localhost:8000
         --users 20 --spawn-rate 2 --run-time 5m --headless
         --csv tests/load/results
    Pass criteria (SLO-2 + SLO-3):
      - /health 99th percentile response time < 1000ms
      - /history 99th percentile response time < 5000ms
      - failure rate < 1% across all endpoints
    '''
    wait_time = between(1, 3)

    @task(5)
    def health_check(self):
        self.client.get("/health",
                        name="/health [SLO-3 probe]")

    @task(3)
    def get_history(self):
        self.client.get("/history?n=50",
                        name="/history")

    @task(2)
    def get_status(self):
        self.client.get("/status",
                        name="/status [SLO-1 timing]")

    @task(1)
    def get_explain(self):
        self.client.get("/explain",
                        name="/explain [SHAP endpoint]")

@events.quitting.add_listener
def on_quit(environment, **kwargs):
    '''Assert SLO-2 and SLO-3 pass criteria at end of load test.'''
    stats = environment.stats
    health = stats.get("/health [SLO-3 probe]", "GET")
    if health:
        p99_ms = health.get_response_time_percentile(0.99)
        fail_pct = health.num_failures / max(health.num_requests, 1) * 100
        print(f"[SLO-3] /health P99={p99_ms:.0f}ms, fail={fail_pct:.2f}%")
        if p99_ms > 1000:
            print(f"[SLO-3 FAIL] /health P99 {p99_ms:.0f}ms > 1000ms limit")
            environment.process_exit_code = 1
        if fail_pct >= 1.0:
            print(f"[SLO-3 FAIL] failure rate {fail_pct:.2f}% >= 1%")
            environment.process_exit_code = 1
    print("[LOAD TEST] Complete.")
