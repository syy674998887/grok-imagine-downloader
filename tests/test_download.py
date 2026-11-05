"""Offline regression tests: python -m unittest discover -s tests -v."""
import io
import json
from pathlib import Path
import plistlib
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import download as downloader

REAL_CHROME_UA = downloader.chrome_ua  # setUp replaces it; its own test needs the real function


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.dest = self.root / "output"
        self.dest.mkdir()
        cookies = self.root / "offline-cookies.txt"
        cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
        self.patch(downloader, "DEST", str(self.dest))
        self.patch(downloader, "COOKIES", str(cookies))
        self.patch(downloader, "UA", "offline-test-UA")
        self.patch(downloader, "chrome_ua", return_value="offline-test-UA")
        self.patch(downloader.shutil, "which", return_value="curl")
        self.patch(downloader.time, "sleep")
        downloader.STOP.clear()
        self.addCleanup(downloader.STOP.clear)
        # Any test that accidentally reaches a real command fails immediately.
        self.process = self.patch(
            downloader.subprocess, "run", side_effect=AssertionError("Unexpected subprocess")
        )
        for stream in ("stdout", "stderr"):
            self.patch(sys, stream, io.StringIO())

    def patch(self, target, name, *args, **kwargs):
        patcher = mock.patch.object(target, name, *args, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def asset(self, asset_id="example-asset", size=4):
        return {
            "assetId": asset_id,
            "mimeType": "video/mp4",
            "fileSource": "IMAGINE_GENERATED_FILE_SOURCE",
            "createTime": "2026-01-02T03:04:05Z",
            "key": asset_id + ".mp4",
            "sizeBytes": size,
            "sourceConversationId": "conversation-1",
            "responseId": asset_id + "-response",
        }

    def item(self, asset_id="example-asset", size=4, payload=None):
        item = downloader.plan([self.asset(asset_id, size)])[0]
        if payload is not None:
            path = Path(item["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        return item

    def metadata(self):
        return json.loads((self.dest / "metadata.json").read_text(encoding="utf-8"))

    @staticmethod
    def page(assets, token=None):
        return json.dumps({"assets": assets, "nextPageToken": token}).encode() + b"200", ""

    @staticmethod
    def serve_pages(pages):
        pages = iter(pages)

        def serve(url, out):
            body, error = next(pages)
            Path(out).write_bytes(body[:-3])
            return body[-3:], error

        return serve

    def test_curl_rejects_partial_http_200_on_transport_failure(self):
        self.process.side_effect = None
        for exit_code in (18, 28):
            with self.subTest(exit_code=exit_code):
                self.process.return_value = subprocess.CompletedProcess(
                    [], exit_code, b"partial body200", b"transfer interrupted"
                )
                body, error = downloader.curl("https://example.invalid/asset")
                self.assertEqual(body[-3:], b"000")
                self.assertIn(str(exit_code), error)
                self.assertIn("transfer interrupted", error)

    def test_curl_preserves_auth_and_rate_limit_statuses(self):
        self.process.side_effect = None
        for status in (b"401", b"403", b"429"):
            with self.subTest(status=status):
                self.process.return_value = subprocess.CompletedProcess([], 22, status, b"HTTP error")
                body, _ = downloader.curl("https://example.invalid/asset")
                self.assertEqual(body[-3:], status)

    def test_curl_preserves_success_body(self):
        self.process.side_effect = None
        self.process.return_value = subprocess.CompletedProcess([], 0, b'{"ok": true}200', b"")
        self.assertEqual(downloader.curl("https://example.invalid/asset"), (b'{"ok": true}200', ""))

    def test_manifest_fetches_beyond_two_hundred_pages(self):
        pages = [self.page([{"assetId": str(n)}], str(n + 1) if n < 201 else None)
                 for n in range(202)]
        request = self.patch(downloader, "curl", side_effect=self.serve_pages(pages))
        assets = downloader.fetch_manifest()
        self.assertEqual(len(assets), 202)
        self.assertEqual(assets[-1]["assetId"], "201")
        self.assertEqual(request.call_count, 202)

    def test_manifest_rejects_repeated_tokens_and_empty_intermediate_pages(self):
        cases = {
            "repeated": [self.page([{}], "a"), self.page([{}], "a")],
            "cycle": [self.page([{}], "a"), self.page([{}], "b"), self.page([{}], "a")],
            "empty": [self.page([], "next")],
        }
        for name, pages in cases.items():
            with self.subTest(name=name), mock.patch.object(downloader, "curl", side_effect=self.serve_pages(pages)):
                with self.assertRaises(RuntimeError):
                    downloader.fetch_manifest()

    def test_manifest_retry_discards_partial_response(self):
        assets = [{"assetId": "retried"}]
        payload = json.dumps({"assets": assets}).encode()

        def transfer(cmd, **kwargs):
            out = cmd[cmd.index("-o") + 1]
            partial = payload[:5]
            if cmd[cmd.index("--retry") + 1] == "0":
                return subprocess.CompletedProcess(cmd, 28, partial + b"200", b"timeout")
            # curl can reset a file between attempts, but stdout keeps the partial body.
            if out == "-":
                body = partial + payload + b"200"
            else:
                Path(out).write_bytes(partial)
                Path(out).write_bytes(payload)
                body = b"200"
            return subprocess.CompletedProcess(cmd, 0, body, b"timeout on first attempt")

        self.process.side_effect = transfer
        self.assertEqual(downloader.fetch_manifest(), assets)

    def test_complete_requires_a_regular_file_and_valid_size(self):
        directory = self.dest / "directory"
        directory.mkdir()
        cases = [
            ({"path": str(directory), "size": 0}, False),
            (self.item("missing", size=0), False),
            (self.item("empty", size=0, payload=b""), False),
            (self.item("unknown", size=0, payload=b"data"), True),
            (self.item("matching", size=4, payload=b"data"), True),
            (self.item("truncated", size=4, payload=b"x"), False),
        ]
        for item, expected in cases:
            with self.subTest(path=item["path"]):
                self.assertEqual(downloader.complete(item), expected)

    def test_download_all_counts_failures_and_keeps_only_complete_files(self):
        existing = self.item("existing", payload=b"data")
        good = self.item("good")
        bad = self.item("bad")

        def transfer(url, out):
            self.assertNotEqual(url, existing["url"])
            Path(out).write_bytes(b"data" if url == good["url"] else b"x")
            return b"200", ""

        self.patch(downloader, "curl", side_effect=transfer)
        self.assertEqual(downloader.download_all([existing, good, bad], workers=2), 1)
        self.assertEqual(Path(good["path"]).read_bytes(), b"data")
        self.assertFalse(Path(bad["path"]).exists())
        self.assertFalse(Path(bad["path"] + ".part").exists())
        self.assertEqual(downloader.download_all([existing, good], workers=2), 0)

    def test_download_does_not_accept_an_empty_unknown_size_file(self):
        item = self.item(size=0)

        def transfer(url, out):
            Path(out).write_bytes(b"")
            return b"200", ""

        self.patch(downloader, "curl", side_effect=transfer)
        self.assertEqual(downloader.download(item)[0], "fail")
        self.assertFalse(Path(item["path"]).exists())

    def test_metadata_keeps_valid_old_files_and_replaces_current_rows(self):
        current = self.item("current", payload=b"data")
        current["gen"] = {"genType": "videoGen", "prompt": "新的提示词"}
        old = []
        for asset_id, size, payload in (
            ("current", 4, b"data"), ("removed", 4, b"data"),
            ("unknown", 0, b"data"), ("truncated", 4, b"x"),
            ("empty", 0, b""), ("missing", 4, None),
        ):
            item = self.item(asset_id, size, payload)
            old.append({"id": asset_id, "file": item["rel"], "size": size, "prompt": "cached"})
        (self.dest / "directory").mkdir()
        old.append({"id": "directory", "file": "directory", "size": 0})
        downloader.write_metadata([current], old)
        rows = {row["id"]: row for row in self.metadata()}
        self.assertEqual(set(rows), {"current", "removed", "unknown"})
        self.assertEqual(rows["current"]["prompt"], "新的提示词")
        self.assertEqual(rows["removed"]["prompt"], "cached")

    def test_prompt_lookup_groups_assets_from_the_same_conversation(self):
        first = self.item("first", payload=b"data")
        second = self.item("second", payload=b"data")
        api = self.patch(downloader, "chat_api", side_effect=[
            ("200", {"responseNodes": [
                {"responseId": first["resp"], "parentResponseId": "human"},
                {"responseId": second["resp"], "parentResponseId": "human"},
            ]}),
            ("200", {"responses": [{"responseId": "human", "mediaGenInput": {
                "videoGen": {"prompt": "A quiet sea", "modelName": "example-model"}
            }}]}),
        ])
        downloader.fetch_prompts([first, second])
        self.assertEqual(api.call_count, 2)
        self.assertEqual(api.call_args_list[1].args[1], {"responseIds": ["human"]})
        self.assertEqual(first["gen"]["prompt"], "A quiet sea")
        self.assertEqual(second["gen"], first["gen"])
        self.assertIn("prompts: found 2/2", sys.stdout.getvalue())

    def test_prompt_lookup_does_not_count_generation_settings_without_a_prompt(self):
        item = self.item(payload=b"data")
        self.patch(downloader, "chat_api", side_effect=[
            ("200", {"responseNodes": [
                {"responseId": item["resp"], "parentResponseId": "human"},
            ]}),
            ("200", {"responses": [{"responseId": "human", "mediaGenInput": {
                "videoGen": {"modelName": "example-model"}
            }}]}),
        ])

        downloader.fetch_prompts([item])

        self.assertIsNone(item["gen"]["prompt"])
        self.assertEqual(item["gen"]["model"], "example-model")
        self.assertIn("prompts: found 0/1", sys.stdout.getvalue())

    def test_main_reuses_prompt_cache_and_skips_existing_files(self):
        item = self.item(payload=b"data")
        downloader.write_metadata([], [{
            "id": item["id"], "file": item["rel"], "size": 4,
            "genType": "videoGen", "prompt": "Cached prompt", "model": "example-model",
        }])
        self.patch(downloader, "fetch_manifest", return_value=[self.asset()])
        lookup = self.patch(downloader, "chat_api", side_effect=AssertionError("Cache was ignored"))
        downloader.main(["--prompts"])
        self.process.assert_not_called()
        lookup.assert_not_called()
        self.assertEqual(self.metadata()[0]["prompt"], "Cached prompt")
        self.assertEqual(self.metadata()[0]["model"], "example-model")

    def test_main_saves_successful_and_previous_metadata_before_failure_exit(self):
        previous = self.item("previous", payload=b"data")
        downloader.write_metadata([previous], [])
        self.patch(downloader, "fetch_manifest", return_value=[self.asset("good"), self.asset("bad")])

        def transfer(url, out):
            if url.endswith("good.mp4"):
                Path(out).write_bytes(b"data")
                return b"200", ""
            return b"500", "simulated failure"

        self.patch(downloader, "curl", side_effect=transfer)
        with self.assertRaises(SystemExit) as result:
            downloader.main(["--workers", "2"])
        self.assertEqual(result.exception.code, 1)
        self.assertEqual({row["id"] for row in self.metadata()}, {"previous", "good"})

    def test_main_merges_partial_generation_metadata_without_losing_cached_fields(self):
        item = self.item(payload=b"data")
        downloader.write_metadata([], [{
            "id": item["id"], "file": item["rel"], "size": 4,
            "genType": "videoGen", "prompt": "Cached prompt", "model": "old-model",
            "aspectRatio": "16:9", "duration": 6, "resolution": "720p",
        }])
        asset = self.asset()
        asset["mediaGenInput"] = {"videoGen": {
            "modelName": "new-model", "prompt": None, "aspectRatio": "",
        }}
        self.patch(downloader, "fetch_manifest", return_value=[asset])

        downloader.main([])

        self.process.assert_not_called()
        self.assertEqual(self.metadata(), [{
            "file": item["rel"], "id": item["id"], "mime": "video/mp4",
            "created": "2026-01-02T03:04:05Z", "size": 4, "w": None, "h": None,
            "genType": "videoGen", "prompt": "Cached prompt", "model": "new-model",
            "aspectRatio": "16:9", "duration": 6, "resolution": "720p",
        }])

    def test_missing_metadata_loads_as_empty(self):
        self.assertEqual(downloader.load_metadata(), [])

    def test_main_does_not_overwrite_invalid_metadata(self):
        listing = self.patch(downloader, "fetch_manifest", return_value=[])
        writer = self.patch(downloader, "write_metadata")
        path = self.dest / "metadata.json"
        for original in ('{"broken":', '{}', '[null]'):
            with self.subTest(original=original):
                path.write_text(original, encoding="utf-8")
                with self.assertRaises(SystemExit) as result:
                    downloader.main([])
                self.assertIn("metadata.json", str(result.exception.code))
                self.assertEqual(path.read_text(encoding="utf-8"), original)
        writer.assert_not_called()
        listing.assert_not_called()  # reported before the (possibly long) asset listing

    def test_main_rejects_nonpositive_workers_before_fetching(self):
        listing = self.patch(downloader, "fetch_manifest")
        for workers in ("0", "-2"):
            with self.subTest(workers=workers), self.assertRaises(SystemExit) as result:
                downloader.main(["--workers", workers])
            self.assertEqual(result.exception.code, 2)
        listing.assert_not_called()

    def test_curl_ends_stalled_transfers_instead_of_capping_their_duration(self):
        self.process.side_effect = None
        self.process.return_value = subprocess.CompletedProcess([], 0, b"200", b"")
        downloader.curl("https://example.invalid/asset", out=str(self.root / "asset"))
        downloader.chat_api("https://example.invalid/chat")
        media, chat = (call.args[0] for call in self.process.call_args_list)
        for cmd in (media, chat):
            self.assertEqual(cmd[cmd.index("--speed-limit") + 1], str(downloader.STALL_RATE))
            self.assertEqual(cmd[cmd.index("--speed-time") + 1], str(downloader.STALL_TIME))
        self.assertEqual(media[media.index("--max-time") + 1], str(downloader.MAX_TIME))
        self.assertEqual(chat[chat.index("--max-time") + 1], "60")

    def test_curl_interrupted_by_ctrl_c_stops_the_run(self):
        self.process.side_effect = None
        for status in (-signal.SIGINT, 0xC000013A):  # POSIX signal, Windows STATUS_CONTROL_C_EXIT
            with self.subTest(status=status):
                downloader.STOP.clear()
                self.process.return_value = subprocess.CompletedProcess([], status, b"", b"")
                self.assertEqual(downloader.curl("https://example.invalid/asset"), (b"000", "interrupted"))
                self.assertTrue(downloader.STOP.is_set())

    def test_download_does_not_retry_after_an_interrupt(self):
        item = self.item()
        self.process.side_effect = None
        self.process.return_value = subprocess.CompletedProcess([], -signal.SIGINT, b"", b"")
        self.assertEqual(downloader.download(item)[0], "cancel")
        self.assertEqual(self.process.call_count, 1)
        self.assertFalse(Path(item["path"] + ".part").exists())

    def test_pool_starts_no_new_work_after_stop(self):
        started, results = [], []

        def work(n):
            started.append(n)
            downloader.STOP.set()
            return n

        downloader.run_pool(work, range(100), 2, results.append)
        self.assertEqual(sorted(results), sorted(started))
        self.assertLessEqual(len(started), 4)  # only the small queued window ran

    def test_main_saves_metadata_and_exits_130_on_ctrl_c(self):
        done = self.item("done", payload=b"data")
        self.patch(downloader, "fetch_manifest", return_value=[self.asset("done"), self.asset("pending")])
        self.patch(downloader, "download_all", side_effect=KeyboardInterrupt)
        with self.assertRaises(SystemExit) as result:
            downloader.main([])
        self.assertEqual(result.exception.code, 130)
        self.assertEqual([row["id"] for row in self.metadata()], [done["id"]])
        self.assertIn("Ctrl+C", sys.stderr.getvalue())

    def test_listing_retries_rate_limited_pages_then_paces_the_rest(self):
        pages = [self.page([{"assetId": "1"}], "a"), (b"429", "rate limited"), (b"429", "rate limited"),
                 self.page([{"assetId": "2"}], "b"), self.page([{"assetId": "3"}])]
        self.patch(downloader, "curl", side_effect=self.serve_pages(pages))
        self.assertEqual([a["assetId"] for a in downloader.fetch_manifest()], ["1", "2", "3"])
        waits = [call.args[0] for call in downloader.time.sleep.call_args_list]
        self.assertEqual(waits, [downloader.LISTING_BACKOFF[0], downloader.LISTING_BACKOFF[1], 1])

    def test_listing_keeps_the_pages_read_before_a_persistent_failure(self):
        attempts = len(downloader.LISTING_BACKOFF) + 1
        pages = [self.page([{"assetId": "1"}], "a")] + [(b"503", "unavailable")] * attempts
        request = self.patch(downloader, "curl", side_effect=self.serve_pages(pages))
        with self.assertRaises(downloader.ListingIncomplete) as result:
            downloader.fetch_manifest()
        self.assertEqual(result.exception.assets, [{"assetId": "1"}])
        self.assertIn("HTTP 503", result.exception.reason)
        self.assertEqual(request.call_count, 1 + attempts)

    def test_listing_does_not_retry_authentication_errors(self):
        request = self.patch(downloader, "curl", side_effect=self.serve_pages([(b"401", "unauthorized")]))
        with self.assertRaises(SystemExit) as result:
            downloader.fetch_manifest()
        self.assertEqual(result.exception.code, 1)
        self.assertEqual(request.call_count, 1)
        self.assertIn("cookie", sys.stderr.getvalue())

    def test_listing_reports_a_page_that_is_not_json(self):
        self.patch(downloader, "curl", side_effect=self.serve_pages([(b"<html>sign in</html>200", "")]))
        with self.assertRaises(SystemExit):
            downloader.fetch_manifest()
        self.assertIn("JSON", sys.stderr.getvalue())

    def test_main_archives_a_partial_listing_and_exits_1(self):
        self.patch(downloader, "fetch_manifest",
                   side_effect=downloader.ListingIncomplete([self.asset("listed")], "page 2: HTTP 429"))

        def transfer(url, out):
            Path(out).write_bytes(b"data")
            return b"200", ""

        self.patch(downloader, "curl", side_effect=transfer)
        with self.assertRaises(SystemExit) as result:
            downloader.main([])
        self.assertEqual(result.exception.code, 1)
        self.assertEqual([row["id"] for row in self.metadata()], ["listed"])
        self.assertIn("page 2: HTTP 429", sys.stderr.getvalue())

    def test_plan_lengthens_the_id_prefix_on_a_name_collision(self):
        items = downloader.plan([self.asset("abcdefgh-1111"), self.asset("abcdefgh-2222")])
        self.assertEqual([Path(item["rel"]).name for item in items],
                         ["20260102-030405_abcdefgh.mp4", "20260102-030405_abcdefgh-222.mp4"])

    def test_plan_skips_malformed_entries_and_neutralises_names(self):
        missing_key = self.asset("missing-key")
        del missing_key["key"]
        bad_size = dict(self.asset("bad-size"), sizeBytes="n/a")
        odd = dict(self.asset("../../x:y"), fileSource="UPLOAD", mimeType="image/x", name="a/../b.e x")
        items = downloader.plan([self.asset("good"), missing_key, bad_size, "not-an-asset", odd])
        self.assertEqual([item["id"] for item in items], ["good", "../../x:y"])
        self.assertEqual(items[1]["rel"], "uploads/20260102-030405_______x_.bin")
        self.assertIn("missing-key, bad-size", sys.stderr.getvalue())

    def test_main_rejects_a_cookie_path_containing_an_equals_sign(self):
        folder = self.root / "a=b"
        folder.mkdir()
        cookies = folder / "cookies.txt"
        cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
        listing = self.patch(downloader, "fetch_manifest")
        with self.assertRaises(SystemExit) as result:
            downloader.main(["--cookies", str(cookies)])
        self.assertIn("'='", str(result.exception.code))
        listing.assert_not_called()

    def test_prompt_lookup_stops_after_a_rate_limit(self):
        first = self.item("first", payload=b"data")
        second = self.item("second", payload=b"data")
        second["conv"] = "conversation-2"
        self.patch(downloader, "PROMPT_WORKERS", 1)
        api = self.patch(downloader, "chat_api", return_value=("429", None))
        downloader.fetch_prompts([first, second])
        self.assertEqual(api.call_count, 1)
        self.assertEqual((first["gen"], second["gen"]), ({}, {}))
        self.assertIn("HTTP 429", sys.stderr.getvalue())

    def test_chrome_ua_uses_the_local_chrome_version(self):
        data, bundle = self.root / "chrome", self.root / "Info.plist"
        data.mkdir()
        self.patch(downloader, "CHROME_DATA", {"darwin": str(data), "win32": str(data)})
        self.patch(downloader, "MAC_CHROME_PLIST", str(bundle))
        cases = [
            ("darwin", "142.0.7444.60\n", None, "(Macintosh; Intel Mac OS X 10_15_7)", "Chrome/142.0.0.0"),
            ("win32", "141.0.1\n", None, "(Windows NT 10.0; Win64; x64)", "Chrome/141.0.0.0"),
            ("darwin", None, "140.0.1", "(Macintosh;", "Chrome/140.0.0.0"),  # app bundle fallback
            ("darwin", "not-a-version", None, None, None),
            ("linux", "142.0\n", None, None, None),
        ]
        for platform, last_version, bundle_version, os_part, chrome in cases:
            with self.subTest(platform=platform, last_version=last_version, bundle_version=bundle_version):
                for path in (data / "Last Version", bundle):
                    if path.exists():
                        path.unlink()
                if last_version is not None:
                    (data / "Last Version").write_text(last_version, encoding="utf-8")
                if bundle_version is not None:
                    with open(bundle, "wb") as f:
                        plistlib.dump({"CFBundleShortVersionString": bundle_version}, f)
                with mock.patch.object(downloader.sys, "platform", platform):
                    ua = REAL_CHROME_UA()
                if chrome is None:
                    self.assertIsNone(ua)
                else:
                    self.assertIn(os_part, ua)
                    self.assertIn(chrome, ua)


if __name__ == "__main__":
    unittest.main()
