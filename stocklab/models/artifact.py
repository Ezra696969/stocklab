"""Kebijakan dan pemeriksaan untuk artifact model yang di-upload.

Modul ini memutuskan berkas apa yang boleh dibuka sama sekali, sebelum satu byte
pun diserahkan ke pustaka pemuat. Urutannya disengaja: berkas yang ditolak di
sini tidak pernah menyentuh ``joblib`` maupun ``onnxruntime``.

Tentang keamanan, dengan jujur
------------------------------
Ada dua kelas format, dan perbedaannya bukan soal derajat:

**ONNX** adalah graf yang di-parse, bukan program. Memuatnya tidak menjalankan
kode dari berkas. Ini jalur yang benar-benar aman, dan karena itu jadi default.

**pickle dan joblib** menjalankan kode arbitrer saat deserialisasi. Itu bukan
bug melainkan cara kerjanya: ``__reduce__`` memang dirancang memanggil apa pun
saat objek disusun ulang. Berkas ``.joblib`` dari sumber yang tidak dipercaya
setara dengan menjalankan skrip asing.

Yang sering diklaim sebagai pengaman — memuat di subprocess — **tidak
menyelesaikan masalah itu**. Subprocess membatasi kerusakan akibat kecelakaan
(model rusak, memori habis, proses menggantung) dan memungkinkan proses dibunuh
saat lewat waktu. Kode jahat di dalamnya tetap berjalan, hanya di proses lain,
dengan hak akses yang sama terhadap berkas dan jaringan user. Di Windows bahkan
batas memori tidak tersedia karena modul ``resource`` tidak ada.

Karena itu pickle dimatikan secara default dan hanya bisa dinyalakan dengan
sadar, dengan peringatan yang menyebut konsekuensinya apa adanya — bukan
disembunyikan di balik kata "sandbox" yang menenangkan tapi tidak akurat.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "ArtifactInfo",
    "ArtifactRejected",
    "LoadPolicy",
    "detect_format",
    "check_policy",
]

ArtifactFormat = Literal["onnx", "joblib", "pickle"]

#: Delapan byte pertama yang menandai format, supaya deteksi tidak bergantung
#: pada ekstensi yang bisa saja dipalsukan atau salah ketik.
_PICKLE_PROTOCOL_MARKERS = (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05")
_ZIP_MAGIC = b"PK\x03\x04"


class ArtifactRejected(Exception):
    """Berkas ditolak sebelum dibuka.

    Pesannya ditujukan langsung ke user, jadi harus menyebut apa yang salah dan
    apa yang bisa dilakukan, bukan sekadar menolak.
    """


@dataclass(frozen=True)
class LoadPolicy:
    """Batas yang berlaku sebelum artifact dibuka."""

    max_bytes: int = 64 * 1024 * 1024
    """Batas ukuran. Model tabular yang wajar jauh di bawah ini; berkas raksasa
    lebih mungkin salah unggah daripada model sungguhan."""

    timeout_seconds: float = 30.0
    """Batas waktu pemuatan dan prediksi di subprocess."""

    allow_pickle: bool = False
    """Izinkan ``.joblib`` dan ``.pkl``.

    Default ``False``. Menyalakannya berarti menerima bahwa berkas yang diunggah
    akan menjalankan kodenya sendiri di mesin ini. Lihat docstring modul.
    """

    def describe_risk(self) -> str:
        """Kalimat yang ditampilkan ke user sebelum mereka memilih."""
        if not self.allow_pickle:
            return (
                "Hanya format ONNX yang diterima. ONNX adalah graf yang dibaca, "
                "bukan program, jadi memuatnya tidak menjalankan kode dari berkas."
            )
        return (
            "Format pickle diizinkan. Memuat berkas .joblib atau .pkl "
            "MENJALANKAN KODE yang ada di dalamnya, dengan hak akses yang sama "
            "seperti aplikasi ini. Subprocess membatasi kerusakan karena "
            "kecelakaan, tapi bukan pengaman terhadap berkas yang dibuat jahat. "
            "Hanya unggah artifact yang Anda buat sendiri."
        )


@dataclass(frozen=True)
class ArtifactInfo:
    """Apa yang berhasil diketahui dari sebuah artifact tanpa memakainya."""

    path: Path
    format: ArtifactFormat
    size_bytes: int
    feature_names: tuple[str, ...] | None = None
    """Fitur yang dibutuhkan model beserta urutannya, kalau tersimpan."""

    n_features: int | None = None
    """Jumlah fitur yang diharapkan, kalau namanya tidak tersimpan."""

    params: dict[str, Any] | None = None
    estimator: str | None = None
    """Nama kelas estimator, sekadar untuk ditampilkan."""

    signal_kind: str = "proba"
    notes: tuple[str, ...] = ()
    """Hal yang perlu diketahui user, misalnya nama fitur yang tidak tersimpan."""

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def executes_code_on_load(self) -> bool:
        return self.format in ("joblib", "pickle")


def detect_format(path: str | Path) -> ArtifactFormat:
    """Tentukan format dari isi berkas, bukan dari ekstensinya.

    Ekstensi adalah klaim pengunggah; byte pertama adalah kenyataan. Berkas
    pickle yang dinamai ``.onnx`` akan tertangkap di sini, bukan saat
    onnxruntime gagal dengan pesan yang membingungkan.
    """
    path = Path(path)
    try:
        with path.open("rb") as handle:
            head = handle.read(16)
    except OSError as error:
        raise ArtifactRejected(f"berkas tidak bisa dibaca: {error}") from None

    if head.startswith(_PICKLE_PROTOCOL_MARKERS):
        return "pickle"
    if head.startswith(_ZIP_MAGIC):
        # joblib menyimpan array besar sebagai zip; pickle biasa tidak.
        return "joblib"
    # ONNX adalah protobuf: bidang pertamanya ir_version, bernomor 1 bertipe
    # varint, sehingga byte pertamanya 0x08.
    if head[:1] == b"\x08":
        return "onnx"

    raise ArtifactRejected(
        "format berkas tidak dikenali. Yang diterima: ONNX (.onnx), atau "
        "artifact scikit-learn (.joblib/.pkl) kalau format pickle diizinkan."
    )


def check_policy(path: str | Path, policy: LoadPolicy) -> ArtifactFormat:
    """Periksa berkas terhadap kebijakan; lempar kalau ditolak.

    Dipanggil sebelum pustaka pemuat mana pun menyentuh berkas.
    """
    path = Path(path)
    if not path.exists():
        raise ArtifactRejected(f"berkas tidak ditemukan: {path}")

    size = path.stat().st_size
    if size == 0:
        raise ArtifactRejected("berkas kosong")
    if size > policy.max_bytes:
        raise ArtifactRejected(
            f"ukuran {size / 1048576:.1f} MB melebihi batas "
            f"{policy.max_bytes / 1048576:.0f} MB"
        )

    fmt = detect_format(path)
    if fmt in ("pickle", "joblib") and not policy.allow_pickle:
        raise ArtifactRejected(
            "berkas ini format pickle, yang menjalankan kodenya sendiri saat "
            "dimuat. Ubah modelmu ke ONNX, atau nyalakan izin pickle secara "
            "sadar kalau artifact ini memang buatanmu sendiri."
        )
    return fmt
