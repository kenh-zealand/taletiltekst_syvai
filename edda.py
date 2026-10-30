#!/usr/bin/env python3
"""
Edda - dansk tale-til-tekst på din egen pc (ingen grafikkort nødvendigt).

Et venligt CLI-lag oven på whisper.cpp og den danske Edda v0.2-model.
Tager lydfiler, videofiler, hele mapper eller YouTube-links og laver
en tekstfil (.txt) og/eller undertekster med tidsstempler (.srt).

Brug:
    python edda.py                                   helt interaktivt
    python edda.py optagelse.m4a                     spørger om model og format
    python edda.py optagelse.m4a -m hurtig -f begge  ingen spørgsmål
    python edda.py D:\\Optagelser -f txt              alle lyd/video-filer i en mappe
    python edda.py https://www.youtube.com/watch?v=XXXXXXXXXXX
    python edda.py møde.mp3 --prompt "Moodle, WISEflow, erhvervsakademi"
    python edda.py D:\\Optagelser -m praecis -f txt --sluk   sluk pc'en bagefter

Mens den transskriberer, kan du trykke:
    P   pause / fortsæt
    S   stop

Kræver ffmpeg på PATH. yt-dlp er kun nødvendig til YouTube-links.
'psutil' (pip install psutil) giver pause/stop-tasterne.
whisper.cpp og modellen hentes automatisk første gang (ca. 0,6-0,9 GB).
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOOL_DIR = Path(__file__).resolve().parent
MODEL_DIR = TOOL_DIR / "models"
DEFAULT_OUTPUT_DIR = TOOL_DIR / "Transskriptioner"

MEDIA_EXTS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".m4v", ".3gp",
}

# Edda v0.2 (whisper-large-v3-turbo, dansk) i whisper.cpp-format.
# Konverteringer lavet af andre end Edda-holdet; vægtene er uændrede (Apache 2.0).
MODELS = {
    "hurtig": {
        "label": "Hurtig  (q5_0, ca. 574 MB - anbefalet)",
        "file": "edda-v0.2-q5_0.bin",
        # (adresse, SHA-256 eller None hvis kilden ikke oplyser en kontrolsum)
        "sources": [
            ("https://huggingface.co/tec-7/edda-v0.2-ggml/resolve/main/edda-v0.2-q5_0.bin",
             "9bdfd79be6fad58b0a6f273526fe7e3fdc856bea7b31c19d032c8352d0bf0197"),
            ("https://huggingface.co/emillykkegrann/edda-v0.2-GGML/resolve/main/ggml-edda-v0.2-q5_0.bin",
             None),
        ],
    },
    "praecis": {
        "label": "Præcis  (q8_0, ca. 874 MB - lidt langsommere)",
        "file": "edda-v0.2-q8_0.bin",
        "sources": [
            ("https://huggingface.co/emillykkegrann/edda-v0.2-GGML/resolve/main/ggml-edda-v0.2-q8_0.bin",
             None),
            ("https://huggingface.co/kstorm77/edda-v0.2/resolve/main/ggml/ggml-edda-v0.2-q8_0.bin",
             "976dd84ac3e4e209d8c77de921d6fd0e42b624470578cd897bef653a97e6b4b7"),
        ],
    },
}

FORMATS = {
    "txt": ("Kun tekst (.txt)", ["-otxt"]),
    "srt": ("Kun undertekster (.srt, grove tidsstempler pr. ca. 30 sek.)", ["-osrt"]),
    "begge": ("Begge dele (.txt + .srt)", ["-otxt", "-osrt"]),
}


# ---------------------------------------------------------------- hjælpere

def find_exe(name: str, required: bool = True):
    path = shutil.which(name)
    if not path and required:
        sys.exit(f"fejl: '{name}' blev ikke fundet på PATH. Installer det og prøv igen.")
    return path


WHISPER_ZIP_URL = ("https://github.com/ggml-org/whisper.cpp/releases/download/"
                   "b5454/whisper-blas-bin-x64.zip")


def _whisper_candidates():
    return (TOOL_DIR / "whisper" / "Release" / "whisper-cli.exe",
            TOOL_DIR / "whisper" / "whisper-cli.exe",
            TOOL_DIR / "whisper" / "Release" / "whisper-cli")


def find_whisper() -> str:
    for candidate in _whisper_candidates():
        if candidate.exists():
            return str(candidate)
    path = shutil.which("whisper-cli")
    if path:
        return path
    if os.name == "nt":
        import zipfile
        print("whisper.cpp er ikke hentet endnu - henter den nu (ca. 22 MB, sker kun én gang).")
        zip_path = TOOL_DIR / "whisper.zip"
        try:
            download(WHISPER_ZIP_URL, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(TOOL_DIR / "whisper")
        except Exception as e:  # noqa: BLE001
            print(f"  kunne ikke hente whisper.cpp: {e}")
        finally:
            zip_path.unlink(missing_ok=True)
        for candidate in _whisper_candidates():
            if candidate.exists():
                return str(candidate)
    sys.exit("fejl: whisper-cli.exe blev ikke fundet i mappen 'whisper\\Release'.\n"
             "Hent 'whisper-blas-bin-x64.zip' fra github.com/ggml-org/whisper.cpp/releases "
             "og pak den ud i 'whisper'-mappen ved siden af edda.py.")


def prompt_choice(prompt: str, options: list[str]) -> int:
    print(f"\n{prompt}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        raw = input(f"Vælg [1-{len(options)}]: ").strip()
        if raw == "":
            return 0
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print("Ugyldigt valg, prøv igen.")


def fmt_duration(seconds) -> str:
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}t {m}m {s}s" if h else f"{m}m {s}s"


def safe_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name[:150] or "transskription"


def default_threads() -> int:
    n = None
    if psutil is not None:
        try:
            n = psutil.cpu_count(logical=False)
        except Exception:
            n = None
    n = n or os.cpu_count() or 4
    return max(1, min(8, n))


def media_duration(path: Path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True)
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


# ---------------------------------------------------------- model-download

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "edda-cli"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last = 0.0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last > 0.3:
                last = now
                if total:
                    pct = done * 100 / total
                    bar = "#" * int(pct / 4)
                    print(f"\r  [{bar:<25}] {pct:5.1f}%  {done / 1e6:6.0f} / {total / 1e6:.0f} MB",
                          end="", flush=True)
                else:
                    print(f"\r  {done / 1e6:6.0f} MB", end="", flush=True)
        print()
    part.replace(dest)


def ensure_model(key: str) -> Path:
    spec = MODELS[key]
    dest = MODEL_DIR / spec["file"]
    if dest.exists():
        return dest
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nModellen '{key}' er ikke hentet endnu - henter den nu (sker kun én gang).")
    for url, expected_sha in spec["sources"]:
        print(f"Kilde: {url}")
        try:
            download(url, dest)
        except Exception as e:  # noqa: BLE001
            print(f"  kunne ikke hente: {e}")
            continue
        if dest.stat().st_size < 100_000_000:
            print("  filen er for lille til at være modellen - prøver næste kilde.")
            dest.unlink(missing_ok=True)
            continue
        if expected_sha:
            print("  tjekker filen ...")
            if sha256_of(dest) != expected_sha:
                print("  filen matcher ikke den forventede kontrolsum - prøver næste kilde.")
                dest.unlink(missing_ok=True)
                continue
        print(f"  gemt: {dest}")
        return dest
    sys.exit("\nfejl: modellen kunne ikke hentes automatisk.\n"
             f"Hent den manuelt fra en af adresserne ovenfor og læg den her:\n  {dest}")


# ------------------------------------------------------------ pause / stop

def _proc_tree(pid):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return []
    procs = [parent]
    try:
        procs += parent.children(recursive=True)
    except psutil.Error:
        pass
    return [p for p in procs if p.is_running()]


def _suspend_tree(pid):
    for p in _proc_tree(pid):
        try:
            p.suspend()
        except psutil.Error:
            pass


def _resume_tree(pid):
    for p in _proc_tree(pid):
        try:
            p.resume()
        except psutil.Error:
            pass


def _kill_tree(pid):
    procs = _proc_tree(pid)
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    if procs:
        _, alive = psutil.wait_procs(procs, timeout=3)
        for p in alive:
            try:
                p.kill()
            except psutil.Error:
                pass


class Status:
    """En statuslinje nederst, der opdateres hvert halve sekund, mens teksten
    løber ind ovenover."""

    SPIN = "|/-\\"

    def __init__(self, audio_seconds=None):
        self.audio_seconds = audio_seconds
        self.phase = "Indlæser model"
        self.pct = None
        self.paused = False
        self.start = time.time()
        self.phase_start = self.start
        self.pause_started = None
        self.paused_total = 0.0
        self.lock = threading.Lock()
        self.tick = 0
        self.width = 0
        self.enabled = sys.stdout.isatty()

    def _elapsed(self):
        now = self.pause_started or time.time()
        return now - self.start - self.paused_total

    def set_phase(self, phase):
        with self.lock:
            if phase != self.phase:
                self.phase = phase
                self.phase_start = time.time()

    def set_pct(self, pct):
        with self.lock:
            self.pct = pct

    def set_paused(self, paused):
        with self.lock:
            if paused and not self.paused:
                self.pause_started = time.time()
            elif not paused and self.paused and self.pause_started:
                self.paused_total += time.time() - self.pause_started
                self.pause_started = None
            self.paused = paused

    def _line(self):
        el = self._elapsed()
        if self.paused:
            return f"  [PAUSE]  {fmt_duration(el)} gået  - tryk P for at fortsætte"
        spin = self.SPIN[self.tick % 4]
        parts = [f"  {spin} {self.phase}"]
        if self.pct is not None:
            filled = int(self.pct / 5)
            parts.append(f"[{'#' * filled}{'.' * (20 - filled)}] {self.pct:3d}%")
        parts.append(f"{fmt_duration(el)} gået")
        if self.pct and 0 < self.pct < 100:
            work_el = time.time() - self.phase_start
            remaining = work_el * (100 - self.pct) / self.pct
            parts.append(f"ca. {fmt_duration(remaining)} tilbage")
        return "  ".join(parts)

    def _write(self, text):
        pad = max(0, self.width - len(text))
        sys.stdout.write("\r" + text + " " * pad)
        sys.stdout.flush()
        self.width = len(text)

    def clear(self):
        if self.enabled and self.width:
            sys.stdout.write("\r" + " " * self.width + "\r")
            self.width = 0

    def print_line(self, text):
        """Skriv en tekstlinje over statuslinjen og tegn statuslinjen igen."""
        with self.lock:
            self.clear()
            print(text, flush=True)
            if self.enabled:
                self._write(self._line())

    def refresh(self):
        with self.lock:
            self.tick += 1
            if self.enabled:
                self._write(self._line())

    def heartbeat_line(self):
        """Når konsollen ikke er interaktiv: en livstegn-linje i ny og næ."""
        with self.lock:
            return self._line().replace(self.SPIN[self.tick % 4] + " ", "")


PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+)%")


def run_with_controls(cmd, log_path: Path, audio_seconds=None) -> tuple[int, bool]:
    """Kør whisper-cli med statuslinje, live-tekst og (hvis muligt) P/S-taster."""
    controls = psutil is not None and msvcrt is not None and sys.stdin.isatty()
    if psutil is None:
        print("(tip: `pip install psutil` giver pause/stop-taster)")
    if controls:
        print("Taster:  [P] pause/fortsæt    [S] stop")
    print("Første tekst kan godt være et minut om at dukke op på en CPU - statuslinjen viser, at den arbejder.\n")

    status = Status(audio_seconds)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    state = {"stopped": False}
    done = threading.Event()

    with open(log_path, "w", encoding="utf-8", errors="replace") as log:

        def read_stderr():
            for line in proc.stderr:
                log.write(line)
                m = PROGRESS_RE.search(line)
                if m:
                    status.set_phase("Transskriberer")
                    status.set_pct(min(100, int(m.group(1))))
                elif "processing" in line and "samples" in line:
                    status.set_phase("Transskriberer")
                    status.set_pct(0)
                elif "loading model" in line or "whisper_model_load" in line:
                    status.set_phase("Indlæser model")

        def read_stdout():
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                status.set_phase("Transskriberer")
                status.print_line(line)

        def ticker():
            last_beat = time.time()
            while not done.wait(0.5):
                if status.enabled:
                    status.refresh()
                elif time.time() - last_beat > 30:
                    last_beat = time.time()
                    print(status.heartbeat_line(), flush=True)

        def listen():
            paused = False
            while not done.is_set():
                try:
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode(errors="ignore").lower()
                        if key == "p":
                            if paused:
                                _resume_tree(proc.pid)
                            else:
                                _suspend_tree(proc.pid)
                            paused = not paused
                            status.set_paused(paused)
                            status.refresh()
                        elif key in ("s", "q"):
                            state["stopped"] = True
                            status.print_line("[Stopper ...]")
                            if paused:
                                _resume_tree(proc.pid)
                            _kill_tree(proc.pid)
                            return
                except (OSError, ValueError):
                    pass
                time.sleep(0.15)

        threads = [threading.Thread(target=f, daemon=True)
                   for f in (read_stderr, read_stdout, ticker)]
        if controls:
            threads.append(threading.Thread(target=listen, daemon=True))
        for t in threads:
            t.start()
        returncode = proc.wait()
        threads[0].join(timeout=2)
        threads[1].join(timeout=2)
        done.set()
        with status.lock:
            status.clear()
        sys.stdout.flush()
    return returncode, state["stopped"]


# ---------------------------------------------------------------- input

def is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://", "www."))


def collect_inputs(raw_inputs: list[str]) -> list[str]:
    items = []
    for raw in raw_inputs:
        raw = raw.strip().strip('"').strip("'").strip()
        if not raw:
            continue
        if is_url(raw):
            items.append(raw)
            continue
        p = Path(raw).expanduser()
        if p.is_dir():
            found = sorted(f for f in p.iterdir() if f.suffix.lower() in MEDIA_EXTS)
            if not found:
                print(f"advarsel: ingen lyd- eller videofiler i {p}")
            items += [str(f) for f in found]
        elif p.is_file():
            items.append(str(p))
        else:
            print(f"advarsel: kan ikke finde '{raw}' - springer over")
    return items


def fetch_youtube_audio(url: str, work: Path) -> tuple[Path, str]:
    ytdlp = find_exe("yt-dlp", required=False)
    if not ytdlp:
        sys.exit("fejl: YouTube-links kræver yt-dlp på PATH.")
    print("Henter info fra YouTube ...")
    info_proc = subprocess.run([ytdlp, "-J", "--no-playlist", url], capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
    title = "youtube"
    if info_proc.returncode == 0:
        try:
            info = json.loads(info_proc.stdout)
            title = info.get("title") or title
            print(f"Titel: {title}")
            if info.get("duration"):
                print(f"Varighed: {fmt_duration(info['duration'])}")
        except json.JSONDecodeError:
            pass
    print("Henter lyden ...")
    proc = subprocess.run([ytdlp, "--no-playlist", "-f", "bestaudio/best",
                           "-o", str(work / "yt.%(ext)s"), url])
    if proc.returncode != 0:
        sys.exit(f"fejl: yt-dlp kunne ikke hente lyden (kode {proc.returncode}).")
    files = [f for f in work.iterdir() if f.name.startswith("yt.")]
    if not files:
        sys.exit("fejl: yt-dlp gav ingen lydfil.")
    return files[0], title


def to_wav(src: Path, dest: Path, ffmpeg: str) -> None:
    proc = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(src),
                           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)])
    if proc.returncode != 0 or not dest.exists():
        raise RuntimeError(f"ffmpeg kunne ikke læse {src.name}")


def unique_base(out_dir: Path, name: str, exts: list[str]) -> Path:
    base = out_dir / name
    n = 2
    while any(base.with_name(base.name + e).exists() for e in exts):
        base = out_dir / f"{name} ({n})"
        n += 1
    return base


# ---------------------------------------------------------------- main

def transcribe_one(item: str, *, whisper: str, ffmpeg: str, model: Path, fmt: str,
                   out_dir: Path, prompt: str, threads: int) -> str:
    """Returnerer 'ok', 'stop' eller 'fejl'."""
    work = Path(tempfile.mkdtemp(prefix="edda_"))
    try:
        if is_url(item):
            src, title = fetch_youtube_audio(item, work)
            name = safe_name(title)
        else:
            src = Path(item)
            name = safe_name(src.stem)
            print(f"Fil: {src.name}")

        # -nt: Edda v0.2 er ikke trænet til Whisper's tidsstempel-tokens; med dem
        # slået til looper afkodningen og ender med tom output.
        # whisper.cpp har problemer med æøå i stier på Windows, så vi
        # arbejder med neutrale filnavne i en temp-mappe og omdøber bagefter.
        wav = work / "lyd.wav"
        print("Klargør lyd (16 kHz mono) ...")
        to_wav(src, wav, ffmpeg)
        dur = media_duration(wav)
        if dur and not is_url(item):
            print(f"Varighed: {fmt_duration(dur)}")

        flags = FORMATS[fmt][1]
        cmd = [whisper, "-m", str(model), "-l", "da", "-t", str(threads), "-pp", "-nt",
               "-f", str(wav), "-of", str(work / "resultat"), *flags]
        if prompt:
            cmd += ["--prompt", prompt]

        print(f"\nTransskriberer med {threads} CPU-tråde ...\n")
        start = time.time()
        rc, stopped = run_with_controls(cmd, work / "whisper.log", dur)
        elapsed = time.time() - start
        if stopped:
            print("\nStoppet.")
            return "stop"
        if rc != 0:
            log = (work / "whisper.log").read_text(encoding="utf-8", errors="replace")
            print(f"\nfejl: whisper-cli stoppede med kode {rc}. Sidste linjer i loggen:")
            print("\n".join(log.strip().splitlines()[-10:]))
            return "fejl"

        # Gem altid den seneste whisper-log + lyd ved siden af værktøjet til fejlsøgning.
        try:
            shutil.copy(work / "whisper.log", TOOL_DIR / "sidste-koersel.log")
        except OSError:
            pass
        exts = [".txt" if f == "-otxt" else ".srt" for f in flags]
        produced_text = "".join(
            (work / f"resultat{e}").read_text(encoding="utf-8", errors="replace")
            for e in exts if (work / f"resultat{e}").exists()).strip()
        if not produced_text:
            try:
                shutil.copy(wav, TOOL_DIR / "sidste-lyd.wav")
            except OSError:
                pass
            log = (work / "whisper.log").read_text(encoding="utf-8", errors="replace")
            print("\nAdvarsel: whisper gav ingen tekst. Sidste linjer i loggen:")
            print("\n".join(log.strip().splitlines()[-15:]))
            print(f"\nHele loggen og lyden er gemt i {TOOL_DIR} (sidste-koersel.log / sidste-lyd.wav).")
        out_dir.mkdir(parents=True, exist_ok=True)
        base = unique_base(out_dir, name, exts)
        saved = []
        for ext in exts:
            produced = work / f"resultat{ext}"
            if produced.exists():
                target = base.with_name(base.name + ext)
                shutil.move(str(produced), str(target))
                saved.append(target)

        speed = f"  ({dur / elapsed:.1f}x realtid)" if dur and elapsed > 0 else ""
        print(f"\nFærdig på {fmt_duration(elapsed)}{speed}.")
        for s in saved:
            print(f"  gemt: {s}")
        return "ok"
    except RuntimeError as e:
        print(f"fejl: {e}")
        return "fejl"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def ask_terms() -> str:
    """Spørg efter fagord - ét eller flere pr. linje, tom linje afslutter."""
    print("\nFagord eller navne der skal staves rigtigt.")
    print("Skriv ét eller flere pr. linje (adskil gerne med komma). Tryk Enter på en tom linje for at gå videre.")
    terms = []
    while True:
        line = input(f"  ord {len(terms) + 1}: " if not terms else "  flere: ").strip()
        if not line:
            break
        terms += [t.strip() for t in line.split(",") if t.strip()]
    if terms:
        print(f"Bruger fagord: {', '.join(terms)}")
    return ", ".join(terms)


def ask_yes_no(question: str) -> bool:
    return input(f"{question} [j/N]: ").strip().lower() in ("j", "ja", "y", "yes")


def countdown_shutdown(seconds: int = 30) -> None:
    """Luk computeren ned efter en nedtælling, medmindre brugeren trykker en tast."""
    can_cancel = msvcrt is not None and sys.stdin.isatty()
    hint = "Tryk en vilkårlig tast for at afbryde." if can_cancel else "Tryk Ctrl+C for at afbryde."
    print(f"\nComputeren lukker ned om {seconds} sekunder. {hint}")
    try:
        for left in range(seconds, 0, -1):
            print(f"\r  Nedlukning om {left:2d} sek. ", end="", flush=True)
            end = time.time() + 1
            while time.time() < end:
                if can_cancel and msvcrt.kbhit():
                    msvcrt.getch()
                    print("\n\nNedlukning afbrudt.")
                    return
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\nNedlukning afbrudt.")
        return
    print("\n\nLukker computeren ned ...")
    if os.name == "nt":
        subprocess.run(["shutdown", "/s", "/t", "0"])
    else:
        subprocess.run(["shutdown", "-h", "now"])


def main():
    parser = argparse.ArgumentParser(
        description="Dansk tale-til-tekst med Edda v0.2 og whisper.cpp (kører på CPU).")
    parser.add_argument("input", nargs="*", help="Lyd/video-fil(er), mappe(r) eller YouTube-link")
    parser.add_argument("-m", "--model", choices=list(MODELS), help="hurtig (standard) eller praecis")
    parser.add_argument("-f", "--format", choices=list(FORMATS), help="txt, srt eller begge")
    parser.add_argument("-o", "--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Hvor filerne gemmes")
    parser.add_argument("--prompt", default=None,
                        help="Fagord/navne der skal staves rigtigt, fx \"Moodle, WISEflow\"")
    parser.add_argument("-t", "--threads", type=int, default=default_threads(), help="Antal CPU-tråde")
    parser.add_argument("--sluk", action="store_true",
                        help="Luk computeren ned, når alt er færdigt (30 sek. til at fortryde)")
    parser.add_argument("--hent-model", action="store_true", help="Hent kun modellen og afslut")
    args = parser.parse_args()

    if args.hent_model:
        ensure_model(args.model or "hurtig")
        return

    whisper = find_whisper()
    ffmpeg = find_exe("ffmpeg")

    fully_interactive = not args.input
    raw_inputs = args.input
    if fully_interactive:
        print("Edda - dansk tale-til-tekst")
        print("Træk en lyd/video-fil eller mappe ind i vinduet, eller indsæt et YouTube-link.")
        raw_inputs = [input("Fil/mappe/link: ")]

    items = collect_inputs(raw_inputs)
    if not items:
        sys.exit("fejl: intet at transskribere.")
    if len(items) > 1:
        print(f"\n{len(items)} filer i køen.")

    if args.model:
        model_key = args.model
    else:
        idx = prompt_choice("Model", [MODELS[k]["label"] for k in MODELS])
        model_key = list(MODELS)[idx]

    if args.format:
        fmt = args.format
    else:
        idx = prompt_choice("Hvad vil du have ud", [FORMATS[k][0] for k in FORMATS])
        fmt = list(FORMATS)[idx]

    prompt = args.prompt
    if prompt is None and fully_interactive:
        prompt = ask_terms()

    shutdown_after = args.sluk
    if not shutdown_after and (not args.model or not args.format):
        shutdown_after = ask_yes_no("\nLuk computeren ned, når transskriptionen er færdig?")
    if shutdown_after:
        print("OK - computeren lukker ned bagefter (du får 30 sek. til at fortryde).")

    model = ensure_model(model_key)
    out_dir = Path(args.output_dir)

    results = {"ok": 0, "fejl": 0}
    for n, item in enumerate(items, 1):
        if len(items) > 1:
            print(f"\n=== {n}/{len(items)} ===")
        res = transcribe_one(item, whisper=whisper, ffmpeg=ffmpeg, model=model, fmt=fmt,
                             out_dir=out_dir, prompt=prompt or "", threads=args.threads)
        if res == "stop":
            if shutdown_after:
                print("Du stoppede selv - computeren lukkes ikke ned.")
            sys.exit(0)
        results[res] += 1

    if len(items) > 1:
        print(f"\nAlt færdigt: {results['ok']} lykkedes, {results['fejl']} fejlede.")
    print(f"Filerne ligger i: {out_dir}")

    if shutdown_after:
        countdown_shutdown(30)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nAfbrudt.")
