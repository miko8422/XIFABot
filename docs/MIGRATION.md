# Migration notes

The original project remains beside this directory at `../../XIFABot`. This clean copy contains the tracked source, configuration, persona assets, tests, and the readable ASR experiment code/results. The package code was moved under `packages/` without changing its internal imports or command entry points.

The copy excludes local environments, runtime logs, generated screenshots, model downloads, `.env` files, and private voice recordings. Restore those locally when setting up the new environment. In particular, the voice runtime needs the same oMLX services, API credentials, and any selected TTS speaker/reference assets as the original project.

The copied configuration keeps `language = "ja"` for Japanese input. No source code was changed in the original project as part of this extraction.
