"""One attempt budget per HTTP operation, rate limiting and resumable backoff."""

from email.utils import parsedate_to_datetime
import math
import time

from anime_recommender.ingestion.yummyanime import YummyAnimeError, YummyAnimeNetworkError


class RunStopped(RuntimeError):
    pass


def retry_after_seconds(value, now):
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else 0.0
    except (ValueError, TypeError):
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - now)
        except (ValueError, TypeError, OverflowError):
            return 0.0


class MetadataRequests:
    def __init__(self, config, state, checkpoint, clock=time.time, sleep=time.sleep):
        self.config, self.state, self.checkpoint = config, state, checkpoint
        self.clock, self.sleep, self.used = clock, sleep, 0
        self.events = []

    def call(self, label, operation):
        for attempt in range(self.config["max_retries"] + 1):
            if self.used >= self.config["max_requests"]:
                raise RunStopped("request_budget")
            now = self.clock()
            wait = max(self.state.get("next_allowed_at", 0),
                       self.state.get("last_request_at", -1e12) + 1 / self.config["requests_per_second"]) - now
            if wait > self.config["max_sleep_seconds"]:
                self.checkpoint()
                raise RunStopped("deferred_rate_limit")
            if wait > 0:
                self.sleep(wait)
            self.used += 1
            self.state["total_requests"] = self.state.get("total_requests", 0) + 1
            self.state["last_request_at"] = self.clock()
            self.checkpoint()  # A crash can replay one in-flight operation, not lose the attempt count.
            try:
                result = operation()
                self.events.append({"operation": label, "http": 200, "attempt": attempt + 1})
                return result
            except YummyAnimeError as error:
                self.events.append({"operation": label, "http": error.status, "attempt": attempt + 1})
                temporary = isinstance(error, YummyAnimeNetworkError) or error.status in {408, 425, 429, 500, 502, 503, 504}
                if temporary:
                    delay = max(self.config["backoff_seconds"] * (2 ** attempt),
                                retry_after_seconds(error.retry_after, self.clock()))
                    self.state["next_allowed_at"] = self.clock() + delay
                    self.checkpoint()
                if not temporary or attempt == self.config["max_retries"]:
                    raise
