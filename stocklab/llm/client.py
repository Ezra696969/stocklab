"""Klien Ollama dengan penurunan mutu yang halus.

Ollama dipilih supaya data portofolio user tidak pernah meninggalkan mesinnya
dan tidak ada biaya API. Konsekuensinya: siapa pun yang membuka proyek ini
kemungkinan besar tidak memasangnya. Panel chat yang melempar exception di mesin
mereka lebih buruk daripada panel chat yang menjelaskan keadaan dan tetap
menampilkan ringkasan.

Karena itu setiap jalur di modul ini punya jawaban, dan tidak ada satu pun yang
melempar ke pemanggil.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from stocklab.llm.context import SYSTEM_PROMPT, fallback_narrative, format_context

__all__ = ["ChatClient", "OllamaStatus", "probe_ollama"]

DEFAULT_MODEL = "llama3.2"

#: Batas waktu satu jawaban. Longgar dengan sengaja: di CPU, panggilan pertama
#: harus memuat model ke memori lebih dulu dan mudah melewati satu menit.
#: Batas yang terlalu ketat membuat pertanyaan pertama user selalu jatuh ke
#: ringkasan template walaupun Ollama sehat -- gagal diam-diam yang terbaca
#: seperti fitur yang tidak berfungsi.
DEFAULT_TIMEOUT = 240.0

#: Berapa lama model ditahan di memori setelah menjawab. Tanpa ini model
#: dilepas setelah beberapa menit menganggur, dan pertanyaan berikutnya
#: membayar ongkos pemuatan lagi.
KEEP_ALIVE = "30m"


@dataclass(frozen=True)
class OllamaStatus:
    """Apa yang bisa dilakukan Ollama di mesin ini, kalau ada."""

    installed: bool
    """Paket python ``ollama`` bisa diimpor."""

    reachable: bool
    """Servisnya menjawab."""

    models: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.installed and self.reachable and bool(self.models)

    def explain(self) -> str:
        """Kalimat yang bisa ditampilkan langsung ke user."""
        if not self.installed:
            return (
                "Paket `ollama` belum terpasang, jadi asisten berjalan dalam mode "
                "ringkasan. Angkanya tetap sama — yang hilang cuma percakapannya."
            )
        if not self.reachable:
            return (
                "Ollama terpasang tapi servisnya tidak menjawab. Jalankan "
                "`ollama serve`, lalu muat ulang halaman ini."
            )
        if not self.models:
            return (
                "Ollama berjalan tapi belum ada model yang diunduh. Coba "
                "`ollama pull llama3.2`."
            )
        return f"Terhubung ke Ollama ({', '.join(self.models[:3])})."


def probe_ollama(timeout: float = 2.0) -> OllamaStatus:
    """Periksa Ollama tanpa pernah melempar error.

    ``timeout`` menjaga aplikasi tetap responsif: servis yang menggantung tidak
    boleh membekukan seluruh halaman hanya karena panel chat ingin menyapa.
    """
    try:
        import ollama
    except ImportError:
        return OllamaStatus(installed=False, reachable=False)

    try:
        client = ollama.Client(timeout=timeout)
        listed = client.list()
    except Exception:
        return OllamaStatus(installed=True, reachable=False)

    names: list[str] = []
    for entry in getattr(listed, "models", None) or listed.get("models", []):
        name = getattr(entry, "model", None) or getattr(entry, "name", None)
        if name is None and isinstance(entry, Mapping):
            name = entry.get("model") or entry.get("name")
        if name:
            names.append(str(name))

    return OllamaStatus(installed=True, reachable=True, models=tuple(names))


class ChatClient:
    """Membungkus Ollama, atau menggantikannya kalau tidak tersedia."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        status: OllamaStatus | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.status = status if status is not None else probe_ollama()
        self.timeout = timeout
        self.model = self._resolve_model(model)

    def _resolve_model(self, preferred: str) -> str:
        """Pakai model yang diminta kalau ada, kalau tidak ambil yang pertama."""
        if not self.status.models:
            return preferred
        for name in self.status.models:
            if name.split(":")[0] == preferred.split(":")[0]:
                return name
        return self.status.models[0]

    @property
    def available(self) -> bool:
        return self.status.usable

    def reply(
        self,
        question: str,
        context: Mapping[str, Any],
        history: Sequence[Mapping[str, str]] = (),
    ) -> str:
        """Jawab satu pertanyaan dengan konteks metrik yang sudah dihitung."""
        if not self.available:
            return self._offline_reply(question, context)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": "Data hasil backtest saat ini:\n\n"
                + format_context(context),
            },
            *[
                {"role": m["role"], "content": m["content"]}
                for m in history
                if m.get("role") in ("user", "assistant")
            ],
            {"role": "user", "content": question},
        ]

        try:
            import ollama

            client = ollama.Client(timeout=self.timeout)
            response = client.chat(
                model=self.model, messages=messages, keep_alive=KEEP_ALIVE
            )
        except Exception as error:
            return (
                f"Ollama gagal menjawab ({type(error).__name__}). "
                f"Ini ringkasan yang dihitung Python:\n\n"
                + fallback_narrative(context)
            )

        content = response.get("message", {}).get("content") if isinstance(response, Mapping) else None
        if content is None:
            content = getattr(getattr(response, "message", None), "content", None)
        return (content or fallback_narrative(context)).strip()

    def _offline_reply(self, question: str, context: Mapping[str, Any]) -> str:
        """Jawaban tanpa LLM: ringkasan yang sama, angka dari sumber yang sama."""
        del question  # tanpa model bahasa, pertanyaannya tidak bisa ditafsirkan
        return fallback_narrative(context)

    def greeting(self, context: Mapping[str, Any]) -> str:
        if not self.available:
            return self.status.explain()
        if not context.get("metrics"):
            return (
                "Siap. Jalankan backtest di panel bawah, lalu tanyakan apa saja "
                "tentang hasilnya."
            )
        return (
            "Backtest sudah selesai. Ada yang ingin ditanyakan tentang hasilnya?\n\n"
            "Jawaban pertama bisa memakan waktu sekitar satu menit karena model "
            "harus dimuat ke memori dulu; sesudah itu jauh lebih cepat."
        )
