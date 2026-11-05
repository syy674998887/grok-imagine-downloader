#!/usr/bin/env python3
"""Archive media returned by Grok's Imagine asset listing via curl and exported cookies."""
import argparse, concurrent.futures, contextlib, json, os, plistlib, re, shutil, signal, subprocess, sys, tempfile
import threading, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor

HOME = os.path.expanduser("~")
ROOT = os.path.dirname(os.path.abspath(__file__))  # project root: the cookie and output/ live here
COOKIES = os.path.join(ROOT, "grok.com_cookies.txt")
DEST = os.path.join(ROOT, "output")
WORKERS = 6
PROMPT_WORKERS = 3  # --prompts lookups stay gentle on the chat API
UA = ""  # set in main(): --ua, else built from the local Chrome version
GEN_KEYS = ("genType", "prompt", "model", "aspectRatio", "duration", "resolution")

EXT = {"video/mp4": ".mp4", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png"}
UNSAFE_NAME_CHARS = re.compile(r"[^\w-]")  # file name parts come from the API: no separators, dots or colons
SAFE_EXT = re.compile(r"\.\w{1,10}")

# A transfer is abandoned when it stalls (below STALL_RATE bytes/s for STALL_TIME s), not after a fixed time:
# a large video on a slow link may need longer than any fixed limit, and each attempt restarts from zero.
# MAX_TIME is only a backstop.
STALL_RATE, STALL_TIME, MAX_TIME = 1024, 60, 3600
# Asset-list pages are retried on rate limits and transient failures, after curl's own two quick retries
# (which already honour Retry-After).
RETRYABLE = ("429", "500", "502", "503", "504", "000")
LISTING_BACKOFF = (5, 15, 30, 60, 120)  # seconds

STOP = threading.Event()  # set on Ctrl+C: no new transfer or retry starts
# curl exit statuses meaning it was interrupted as well: killed by SIGINT/SIGTERM (POSIX) or
# STATUS_CONTROL_C_EXIT (Windows, where Ctrl+C reaches every process attached to the console).
INTERRUPTED = {-signal.SIGINT, -signal.SIGTERM, 0xC000013A}

# Chrome's reduced UA: only the major version changes, the platform part is frozen.
UA_FMT = "Mozilla/5.0 ({}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{}.0.0.0 Safari/537.36"
UA_OS = {"darwin": "Macintosh; Intel Mac OS X 10_15_7", "win32": "Windows NT 10.0; Win64; x64"}
CHROME_DATA = {"darwin": os.path.join(HOME, "Library", "Application Support", "Google", "Chrome"),
               "win32": os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "User Data")}
MAC_CHROME_PLIST = "/Applications/Google Chrome.app/Contents/Info.plist"


class Abort(RuntimeError):
    """A problem the user can fix: main() prints it and exits 1 without a traceback."""


class ListingIncomplete(RuntimeError):
    """The asset listing ended early; .assets holds the pages that were read."""

    def __init__(self, assets, reason):
        super().__init__(reason)
        self.assets, self.reason = assets, reason


def chrome_ua():
    """UA of the local Chrome, from the version it last ran as; None if not found."""
    if sys.platform not in UA_OS:
        return None
    ver = ""
    with contextlib.suppress(OSError):
        with open(os.path.join(CHROME_DATA[sys.platform], "Last Version"), encoding="utf-8") as f:
            ver = f.read()
    if not ver and sys.platform == "darwin":  # fall back to the installed app bundle
        with contextlib.suppress(OSError, ValueError):
            with open(MAC_CHROME_PLIST, "rb") as f:
                ver = plistlib.load(f).get("CFBundleShortVersionString", "")
    major = ver.strip().split(".")[0]
    return UA_FMT.format(UA_OS[sys.platform], major) if major.isdigit() else None


def curl(url, out=None, timeout=None, data=None, retry=2):
    """Run curl. Returns (stdout ending with the 3-digit HTTP status, stderr text).
    timeout=None means no fixed limit: only stall detection and the MAX_TIME backstop apply."""
    # -f: no body on HTTP errors, so a retried 429/5xx can't leave an error page in front of the real output.
    # No --compressed: older Windows 10 curl.exe is built without zlib and rejects that option.
    # Only long-standing options are used, so older curl.exe builds on Windows 10 keep working.
    cmd = ["curl", "-fsS", "-b", COOKIES, "-A", UA, "-H", "referer: https://grok.com/",
           "--connect-timeout", "30", "--speed-limit", str(STALL_RATE), "--speed-time", str(STALL_TIME),
           "--max-time", str(timeout or MAX_TIME), "--retry", str(retry), "--retry-delay", "2",
           "-w", "%{http_code}", url]
    if data is not None:  # JSON body via stdin (no Windows command-line length limit) -> POST
        cmd += ["-H", "content-type: application/json", "--data-binary", "@-"]
    cmd += ["-o", out] if out else ["-o", "-"]
    r = subprocess.run(cmd, input=None if data is None else json.dumps(data).encode(), capture_output=True)
    err = r.stderr.decode("utf-8", "replace")
    if r.returncode in INTERRUPTED:  # Ctrl+C reached curl too: stop the run instead of retrying
        STOP.set()
        return b"000", "interrupted"
    if r.returncode:
        # A server can send HTTP 200 before a transfer fails. Never accept a partial body.
        # Keep HTTP errors so callers can distinguish expired cookies and rate limits.
        code = r.stdout[-3:]
        status = code if code.isdigit() and int(code) >= 400 else b"000"
        return status, f"curl exited with code {r.returncode}: {err}"
    return r.stdout, err


def fetch_page(url, page):
    """Fetch one asset-list page, retrying rate limits and transient failures with growing pauses.
    Returns (data, code, err, throttled); data is None if the page could not be read."""
    throttled, waits = False, list(LISTING_BACKOFF)
    while True:
        # curl can truncate an output file on retry, but cannot rewind stdout.
        with tempfile.TemporaryDirectory() as tmp:
            response_path = os.path.join(tmp, "manifest.json")
            body, err = curl(url, out=response_path)
            code = body[-3:].decode()
            throttled = throttled or code == "429" or "error: 429" in err  # curl may have retried a 429 itself
            if code == "200":
                try:
                    with open(response_path, "rb") as f:
                        data = json.load(f)
                except ValueError:
                    data = None
                if isinstance(data, dict):
                    return data, code, err, throttled
                return None, code, "the response is not the expected JSON (a sign-in or challenge page?)", throttled
        if STOP.is_set():
            raise KeyboardInterrupt
        if code not in RETRYABLE or not waits:
            return None, code, err, throttled
        wait = waits.pop(0)
        print(f"  manifest page {page}: HTTP {code}, retrying in {wait}s", file=sys.stderr)
        time.sleep(wait)  # main thread: Ctrl+C interrupts this sleep on every platform


def fetch_manifest():
    """Page through /rest/assets until nextPageToken runs out.
    If the listing ends early after the first page, raises ListingIncomplete with the assets read so far."""
    all_assets, token, page, seen_tokens, pacing = [], "", 0, set(), 0
    while True:
        url = ("https://grok.com/rest/assets?pageSize=100"
               "&orderBy=ORDER_BY_CREATE_TIME&workspaceKind=WORKSPACE_KIND_IMAGINE_ALL")
        if token:
            url += "&pageToken=" + urllib.parse.quote(token, safe="")
        data, code, err, throttled = fetch_page(url, page + 1)
        if throttled:
            pacing = 1  # Grok is rate limiting the listing: pause between the remaining pages
        if data is None:
            print(f"  manifest page {page+1}: HTTP {code} {err}", file=sys.stderr)
            if code in ("401", "403"):
                print(f"  cookie 已失效，或 UA 与导出 cookie 的浏览器不一致（当前 UA: {UA}）。\n"
                      "  重新导出 cookie；Chrome 刚升级过就先重启 Chrome 再导出；"
                      "不是 Chrome 的话用 --ua 传入浏览器控制台里 navigator.userAgent 的值。", file=sys.stderr)
            elif code == "000":
                print("  请求未完成或传输失败，请查看上面的 curl 错误。需要代理时先设置 HTTPS_PROXY 环境变量。",
                      file=sys.stderr)
            if not all_assets:
                sys.exit(1)
            raise ListingIncomplete(all_assets, f"page {page+1}: HTTP {code}")
        batch = data.get("assets") or []
        token = data.get("nextPageToken") or ""
        if not isinstance(batch, list) or not isinstance(token, str):
            raise ListingIncomplete(all_assets, f"page {page+1}: unexpected response format")
        all_assets += batch
        page += 1
        print(f"  page {page}: +{len(batch)} (total {len(all_assets)})")
        if not token:
            return all_assets
        if not batch:
            raise ListingIncomplete(all_assets, f"page {page} was empty but pointed to a next page")
        if token in seen_tokens:
            raise ListingIncomplete(all_assets, f"page {page} repeated an earlier page token")
        seen_tokens.add(token)
        if pacing:
            time.sleep(pacing)


def gen_info(a):
    """Generation params; mediaGenInput is nested as {genType: {prompt, modelName, ...}}."""
    mgi = a.get("mediaGenInput")
    if not isinstance(mgi, dict) or not mgi:
        return {}
    kind, inner = next(iter(mgi.items()))
    if not isinstance(inner, dict):
        return {"genType": kind}
    return {
        "genType": kind,
        "prompt": inner.get("prompt") or None,
        "model": inner.get("modelName") or None,
        "aspectRatio": inner.get("aspectRatio") or None,
        "duration": inner.get("duration") or None,
        "resolution": inner.get("resolutionName") or None,
    }


def plan(assets):
    """Map each asset to a destination path; dedupe filename collisions.
    Entries without the required fields are skipped with a warning instead of aborting the run."""
    used, seen, items, skipped = set(), set(), [], []
    for a in assets:
        try:
            if a.get("isDeleted"):
                continue
            aid, mime, created, key = a["assetId"], a["mimeType"], a["createTime"], a["key"]
            if not all(isinstance(v, str) and v for v in (aid, mime, created, key)):
                raise ValueError("a required field is missing or not a string")
            size = int(a.get("sizeBytes") or 0)
            if size < 0:
                raise ValueError("negative size")
        except (AttributeError, KeyError, TypeError, ValueError):
            skipped.append(a.get("assetId") if isinstance(a, dict) else None)
            continue
        if aid in seen:
            continue
        seen.add(aid)
        source_name = a.get("name")
        ext = EXT.get(mime) or (os.path.splitext(source_name)[1] if isinstance(source_name, str) else "") or ".bin"
        if not SAFE_EXT.fullmatch(ext):
            ext = ".bin"
        gen = a.get("fileSource") == "IMAGINE_GENERATED_FILE_SOURCE"
        sub = "uploads" if not gen else ("video" if mime == "video/mp4" else "image")
        # Unchanged for well-formed data (UTC timestamps, UUIDs), so existing archives keep their file names.
        ts = UNSAFE_NAME_CHARS.sub("_", created.replace("-", "").replace(":", "").replace("T", "-")[:15])
        sid = UNSAFE_NAME_CHARS.sub("_", aid)
        n = 8
        while True:
            name = f"{ts}_{sid[:n]}{ext}"
            rel = f"{sub}/{name}"  # always "/", so metadata.json is the same on Mac and Windows
            if rel not in used or n >= len(sid):
                break
            n += 4
        k = 2
        while rel in used:  # only possible when one ID is a prefix of another
            name = f"{ts}_{sid}_{k}{ext}"
            rel, k = f"{sub}/{name}", k + 1
        used.add(rel)
        items.append({
            "url": "https://assets.grok.com/" + key,
            "path": os.path.join(DEST, sub, name),
            "rel": rel,
            "size": size,
            "id": aid,
            "mime": mime,
            "created": created,
            "w": a.get("width"), "h": a.get("height"),
            "gen": gen_info(a),
            "conv": a.get("sourceConversationId"), "resp": a.get("responseId"),  # for --prompts
        })
    if skipped:
        shown = ", ".join(str(s) for s in skipped[:5]) + (" ..." if len(skipped) > 5 else "")
        print(f"  跳过 {len(skipped)} 个缺少必要字段的资源条目: {shown}", file=sys.stderr)
    return items


def load_metadata():
    """Load the prompt cache; refuse to overwrite an existing file that cannot be read."""
    try:
        with open(os.path.join(DEST, "metadata.json"), encoding="utf-8-sig") as f:  # -sig: tolerate a BOM
            rows = json.load(f)
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        raise Abort(f"cannot read metadata.json ({e}); restore or back it up before rebuilding it") from e
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise Abort("metadata.json must contain a list of objects; the existing file was not changed")
    return rows


def chat_api(url, data=None):
    """One request to a grok.com chat endpoint, no retries; returns (http_code, json or None)."""
    body, _ = curl(url, timeout=60, data=data, retry=0)
    code = body[-3:].decode()
    try:
        return code, json.loads(body[:-3]) if code == "200" else None
    except ValueError:
        return code, None


def pause(seconds, step=0.25):
    """Sleep in a worker thread, ending early on Ctrl+C. Returns True if the run is stopping."""
    for _ in range(max(1, round(seconds / step))):
        if STOP.is_set():
            return True
        time.sleep(step)
    return STOP.is_set()


def run_pool(fn, items, workers, handle):
    """Run fn over items in a thread pool and pass each result to handle() as soon as it is ready.

    Only a small window of work is queued, so each wait stays cheap for large libraries and nothing new
    starts once STOP is set. The short wait timeout keeps Ctrl+C responsive on Windows, where an untimed
    wait cannot be interrupted. On Ctrl+C or an error, queued work is dropped, running work ends at its
    next STOP check, and the exception propagates."""
    todo, end = iter(items), object()
    ex = ThreadPoolExecutor(max_workers=workers)
    pending = set()
    try:
        while True:
            while len(pending) < 2 * workers and not STOP.is_set():
                item = next(todo, end)
                if item is end:
                    break
                pending.add(ex.submit(fn, item))
            if not pending:
                break
            done, pending = concurrent.futures.wait(pending, timeout=0.5,
                                                    return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                handle(f.result())
    except BaseException:
        STOP.set()
        for f in pending:
            f.cancel()
        raise
    finally:
        ex.shutdown(wait=True)


def fetch_prompts(items):
    """--prompts: when the asset list omits mediaGenInput, try the message that started each
    generation. Per source conversation: response-node (who replied to what), then
    load-responses for the parent human messages. Stops on 401/403/429; downloads are unaffected."""
    todo = {}
    for it in items:
        if not it["gen"] and it["conv"] and it["resp"] and not it["rel"].startswith("uploads/") and complete(it):
            todo.setdefault(it["conv"], []).append(it)
    if not todo:
        return
    total = sum(len(v) for v in todo.values())
    print(f"\nLooking up prompts: {total} files in {len(todo)} conversations...")
    stopped = []

    def lookup(conv):
        if stopped or STOP.is_set():
            return 0
        base = f"https://grok.com/rest/app-chat/conversations/{conv}"
        code, nodes = chat_api(base + "/response-node?includeThreads=true")
        if code in ("401", "403", "429"):
            stopped.append(code)
        if stopped or STOP.is_set() or not nodes:
            return 0
        try:
            parent = {n.get("responseId"): n.get("parentResponseId") for n in nodes.get("responseNodes", [])}
            want = sorted({parent.get(it["resp"]) for it in todo[conv]} - {None})
            if not want:
                return 0
            code, resp = chat_api(base + "/load-responses", {"responseIds": want})
            if code in ("401", "403", "429"):
                stopped.append(code)
            msgs = {r.get("responseId"): r for r in (resp or {}).get("responses", [])}
            found = 0
            for it in todo[conv]:
                it["gen"] = gen_info(msgs.get(parent.get(it["resp"]), {}))
                found += bool(it["gen"].get("prompt"))
            return found
        except (AttributeError, TypeError):  # unexpected response shape: skip this conversation
            return 0

    found = []
    run_pool(lookup, list(todo), PROMPT_WORKERS, found.append)
    print(f"  prompts: found {sum(found)}/{total}")
    if stopped:
        print(f"  查询提示词时遇到 HTTP {stopped[0]}，已停止查询。已查到的会保存，稍后再加 --prompts 运行即可补上。",
              file=sys.stderr)


def complete(it):
    """File is on disk with the byte count the API reported (0 = unknown)."""
    p = it["path"]
    if not os.path.isfile(p):
        return False
    size = os.path.getsize(p)
    return size == it["size"] if it["size"] else size > 0


def download(it):
    """Fetch one asset, verifying byte count. Returns (status, item, detail)."""
    p, want = it["path"], it["size"]
    if complete(it):
        return ("skip", it, os.path.getsize(p))
    tmp, detail = p + ".part", "interrupted"
    for attempt in range(3):
        if STOP.is_set() or (attempt and pause(1.5 * attempt)):  # no new attempt after Ctrl+C
            break
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            body, err = curl(it["url"], out=tmp)
            code = body.decode(errors="replace")[-3:]
            got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            if code == "200" and (got == want if want else got > 0):
                os.replace(tmp, p)
                return ("ok", it, got)
            detail = f"http={code} got={got} want={want} {err[:120]}".strip()
        except OSError as e:  # e.g. on Windows an antivirus scan can briefly lock the .part file
            detail = f"{type(e).__name__}: {e}"
    with contextlib.suppress(OSError):
        os.remove(tmp)
    return ("cancel" if STOP.is_set() else "fail", it, detail)


def download_all(items, workers):
    total = sum(i["size"] for i in items)
    print(f"\n{len(items)} assets, {total/1048576:.1f} MB -> {DEST}\n")
    done = {"ok": 0, "skip": 0, "fail": 0, "cancel": 0}
    fails, got_bytes, t0 = [], [0], time.time()

    def handle(result):
        st, it, detail = result
        done[st] += 1
        if st == "fail":
            fails.append((it["rel"], detail))
        elif st != "cancel":
            got_bytes[0] += detail
        n = sum(done.values())
        if n % 25 == 0 or n == len(items):
            el = time.time() - t0
            print(f"  [{n}/{len(items)}] ok={done['ok']} skip={done['skip']} "
                  f"fail={done['fail']}  {got_bytes[0]/1048576:.0f} MB  {el:.0f}s")

    try:
        run_pool(download, items, workers, handle)
    finally:  # also after Ctrl+C; items that did not finish count as cancelled
        left = len(items) - done["ok"] - done["skip"] - done["fail"]
        cancelled = f" cancelled={left}" if left else ""
        print(f"\nDone: ok={done['ok']} skip={done['skip']} fail={done['fail']}{cancelled}")
        if fails:
            print("\nFailures:")
            for rel, d in fails:
                print(f"  {rel}: {d}")
    return done["fail"]


def write_json(path, data):
    """UTF-8 + LF regardless of OS locale; written to a temp file first so it is never half-written."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def write_metadata(items, old):
    """Sidecar metadata (prompts + dimensions) for the files on disk."""
    out = [{"file": i["rel"], "id": i["id"], "mime": i["mime"], "created": i["created"],
            "size": i["size"], "w": i["w"], "h": i["h"], **i["gen"]}
           for i in items if complete(i)]
    ids = {r["id"] for r in out}
    # Keep earlier rows for complete local files, including assets no longer listed by Grok.
    out += [r for r in old if r.get("id") not in ids and r.get("file")
            and complete({"path": os.path.join(DEST, r["file"]), "size": r.get("size") or 0})]
    os.makedirs(DEST, exist_ok=True)
    p = os.path.join(DEST, "metadata.json")
    write_json(p, out)

    kinds = {}
    for r in out:
        sub = r["file"].split("/")[0]
        kinds[sub] = kinds.get(sub, 0) + 1
    withp = sum(1 for r in out if r.get("prompt"))
    print(f"\nmetadata.json: {len(out)} 条, 带 prompt {withp} 条 -> {p}")
    print("分类:", json.dumps(kinds, ensure_ascii=False))


def archive(args):
    if not shutil.which("curl"):
        sys.exit("curl not found (built into macOS and Windows 10+)")
    if not os.path.exists(COOKIES):
        sys.exit(f"cookie file missing: {COOKIES}\n把导出的 cookie 存成这个文件，或用 --cookies 指定路径")
    if "=" in COOKIES:  # curl -b treats an argument containing "=" as cookie data, not as a file name
        sys.exit(f"cookie 文件路径不能包含 '='（curl 会把整个路径当作 cookie 内容发送）。请移动或重命名：{COOKIES}")
    if not UA:
        sys.exit("找不到本机 Chrome：用 --ua 传入导出 cookie 的浏览器的 UA（浏览器控制台运行 navigator.userAgent）")
    print(f"UA: {UA}")
    old = load_metadata()  # before any request: a damaged file is reported before a long listing
    print("Fetching manifest...")
    problem = None
    try:
        assets = fetch_manifest()
    except ListingIncomplete as e:  # archive what was listed rather than nothing
        assets, problem = e.assets, e.reason
        print(f"  资源清单未能读完（{problem}），先处理已列出的 {len(assets)} 个条目。", file=sys.stderr)
    items = plan(assets)
    cache = {r["id"]: {k: r[k] for k in GEN_KEYS if k in r} for r in old if r.get("genType") and r.get("id")}
    for it in items:
        it["gen"] = {**cache.get(it["id"], {}),
                     **{k: v for k, v in it["gen"].items() if v is not None and v != ""}}
    try:
        failures = download_all(items, args.workers)
        if args.prompts and not STOP.is_set():
            fetch_prompts(items)
    finally:  # also on Ctrl-C or an error: keep what this run already found
        write_metadata(items, old)
    if STOP.is_set():  # curl was interrupted although this process saw no KeyboardInterrupt
        raise KeyboardInterrupt
    if problem:
        print(f"\n资源清单不完整（{problem}）：本次只处理了已列出的资源，稍后重新运行即可补齐。", file=sys.stderr)
    if failures or problem:
        sys.exit(1)


def main(argv=None):
    global COOKIES, DEST, UA
    if hasattr(sys.stdout, "reconfigure"):  # Windows files/pipes use the ANSI code page: print '?' instead of crashing
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="下载 Grok Imagine 资产列表返回的媒体并保存元数据（macOS / Windows）")
    ap.add_argument("--cookies", default=COOKIES, help="Netscape 格式的 cookie 文件（默认 %(default)s）")
    ap.add_argument("--dest", default=DEST, help="下载目录（默认 %(default)s）")
    ap.add_argument("--ua", help="导出 cookie 的浏览器的 UA（默认按本机 Chrome 版本生成）")
    ap.add_argument("--workers", type=int, default=WORKERS, help="并发数（默认 %(default)s）")
    ap.add_argument("--prompts", action="store_true",
                    help="到生成文件的来源对话里查提示词等参数（每段对话最多 2 个请求，复用已保存参数）")
    args = ap.parse_args(argv)
    if args.workers < 1:
        ap.error("--workers must be at least 1")
    COOKIES, DEST = os.path.expanduser(args.cookies), os.path.expanduser(args.dest)  # "--dest=~/x" is not expanded by shells
    UA = args.ua or chrome_ua()
    STOP.clear()
    try:
        archive(args)
    except KeyboardInterrupt:  # metadata was already written by archive()'s cleanup
        STOP.set()
        print("\n已中断（Ctrl+C）。已完成的文件会保留，重新运行会跳过它们。", file=sys.stderr)
        sys.exit(130)
    except Abort as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
