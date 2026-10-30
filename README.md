# Edda – dansk tale-til-tekst

Et lille CLI-værktøj, der transskriberer dansk tale lokalt på din pc med
[Edda v0.2](https://huggingface.co/danish-foundation-models/edda-v0.2)
(en dansk Whisper-large-v3-turbo) og [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
Det kører på CPU'en, så du behøver ikke et grafikkort, og ingen lyd forlader maskinen.

Værktøjet tager lyd- og videofiler, hele mapper eller YouTube-links og giver
dig ren tekst (`.txt`), undertekster med grove tidsstempler (`.srt`) eller begge dele.

## Det skal du have

- **Python 3.10+**
- **ffmpeg** på PATH
- **yt-dlp** på PATH. Det er kun nødvendigt til YouTube-links.
- **psutil** (`pip install psutil`), hvis du vil kunne pause og stoppe.
- whisper.cpp (b5454, OpenBLAS-build til Windows x64) hentes automatisk ned i `whisper\` første gang (ca. 22 MB).
- Modellen, som hentes automatisk ned i `models\` første gang (574 MB for *hurtig*).

## Installation

```
git clone https://github.com/kenh-zealand/taletiltekst_syvai
cd taletiltekst_syvai
pip install psutil
```

Ved første kørsel henter værktøjet selv whisper.cpp og modellen. Det sker kun én gang.

## Brug

Den nemmeste måde er at **dobbeltklikke på `edda.bat`**. Du kan også **trække
en lydfil eller en mappe ind på `edda.bat`**.

Fra kommandolinjen:

```
python edda.py                                    helt interaktivt
python edda.py optagelse.m4a                      spørger om model og format
python edda.py optagelse.m4a -m hurtig -f begge   ingen spørgsmål
python edda.py D:\Optagelser -f txt               hele mappen
python edda.py https://www.youtube.com/watch?v=XXXXXXXXXXX
python edda.py møde.mp3 --prompt "Moodle, WISEflow, erhvervsakademi"
python edda.py D:\Optagelser -m praecis -f txt --sluk   sluk pc'en, når den er færdig
python edda.py --hent-model -m praecis            hent kun modellen på forhånd
```

## Valgmuligheder

| Flag | Betydning |
|---|---|
| `-m, --model {hurtig,praecis}` | `hurtig` = q5_0, ca. 574 MB, anbefalet · `praecis` = q8_0, ca. 874 MB, lidt langsommere |
| `-f, --format {txt,srt,begge}` | Hvad der skal laves |
| `-o, --output-dir` | Hvor filerne gemmes (standard undermappen `Transskriptioner` ved siden af værktøjet) |
| `--prompt "..."` | Fagord og navne, som modellen skal stave rigtigt |
| `-t, --threads` | Antal CPU-tråde (standard: antal fysiske kerner, højst 8) |
| `--sluk` | Lukker computeren ned, når alt er færdigt. Du får 30 sek. til at fortryde med en vilkårlig tast |
| `--hent-model` | Henter modellen og afslutter |

Resultatet får samme navn som kildefilen eller YouTube-titlen, fx
`Møde 12. okt.txt`. Hvis navnet er optaget, tilføjes et nummer.

## Mens den transskriberer

| Tast | Effekt |
|---|---|
| `P` | Pause / fortsæt |
| `S` (eller `Q`) | Stop |

Teksten vises løbende med tidsstempler, efterhånden som den bliver klar.

## Noter

- Den er udelukkende til dansk (`-l da`).
- Tidsstempler er slået fra i whisper (`-nt`), fordi Edda v0.2 ellers looper og giver tom
  tekst. `.srt`-filer får derfor kun grove tidsstempler (et afsnit pr. ca. 30 sekunder).
- `--prompt` hjælper på stavningen af navne og fagtermer. Edda v0.2 er trænet til at bruge sådan en prompt.
- Lange optagelser kører i ét hug. Du behøver ikke dele dem op.
- whisper.cpp kan have problemer med æøå i filstier på Windows. Derfor arbejder
  værktøjet med neutrale filnavne i en temp-mappe og omdøber resultatet bagefter.
- Model-filerne er konverteringer lavet af andre end Edda-holdet (tec-7, kstorm77,
  emillykkegrann på Hugging Face). Vægtene er uændrede, og licensen er Apache 2.0.
  Downloads tjekkes med SHA-256. Kreditér den oprindelige Edda-model fra
  Alexandra Instituttet / CoRal / Danish Foundation Models.
- Kun til lyd, du har ret til at transskribere.
