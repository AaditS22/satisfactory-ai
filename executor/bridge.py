import json
import urllib.error
import urllib.request


class BridgeError(RuntimeError):
    """"""


class Bridge:
    def __init__(self, host="127.0.0.1", port=18642, timeout=10.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def request(self, method, path, body=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status, raw = resp.status, resp.read()
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read()
        except urllib.error.URLError as e:
            raise BridgeError(f"cannot reach game at {self.base} ({e.reason}).") from e

        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise BridgeError(f"HTTP {status}, non-JSON reply: {raw[:200]!r}") from e

        if not payload.get("ok"):
            if payload.get("errorCode", "").endswith("route_handler_not_found"):
                raise BridgeError(
                    f"no route for {method} {path}: unknown command, "
                    "or the game is at the main menu with no save loaded"
                )
            raise BridgeError(f"HTTP {status}: {payload.get('error', payload)}")
        return payload

    def ping(self):
        return self.request("GET", "/ping")


if __name__ == "__main__":
    print(Bridge().ping())