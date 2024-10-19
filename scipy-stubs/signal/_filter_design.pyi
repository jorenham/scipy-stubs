from scipy._typing import Untyped

__all__ = [
    "BadCoefficients",
    "band_stop_obj",
    "bessel",
    "besselap",
    "bilinear",
    "bilinear_zpk",
    "buttap",
    "butter",
    "buttord",
    "cheb1ap",
    "cheb1ord",
    "cheb2ap",
    "cheb2ord",
    "cheby1",
    "cheby2",
    "ellip",
    "ellipap",
    "ellipord",
    "findfreqs",
    "freqs",
    "freqs_zpk",
    "freqz",
    "freqz_zpk",
    "gammatone",
    "group_delay",
    "iircomb",
    "iirdesign",
    "iirfilter",
    "iirnotch",
    "iirpeak",
    "lp2bp",
    "lp2bp_zpk",
    "lp2bs",
    "lp2bs_zpk",
    "lp2hp",
    "lp2hp_zpk",
    "lp2lp",
    "lp2lp_zpk",
    "normalize",
    "sos2tf",
    "sos2zpk",
    "sosfreqz",
    "tf2sos",
    "tf2zpk",
    "zpk2sos",
    "zpk2tf",
]

EPSILON: float

filter_dict: Untyped
band_dict: Untyped
bessel_norms: Untyped

class BadCoefficients(UserWarning): ...

def findfreqs(num: Untyped, den: Untyped, N: Untyped, kind: str = "ba") -> Untyped: ...
def freqs(b: Untyped, a: Untyped, worN: int = 200, plot: Untyped | None = None) -> Untyped: ...
def freqs_zpk(z: Untyped, p: Untyped, k: Untyped, worN: int = 200) -> Untyped: ...
def freqz(
    b: Untyped,
    a: int = 1,
    worN: int = 512,
    whole: bool = False,
    plot: Untyped | None = None,
    fs: Untyped = ...,
    include_nyquist: bool = False,
) -> Untyped: ...
def freqz_zpk(z: Untyped, p: Untyped, k: Untyped, worN: int = 512, whole: bool = False, fs: Untyped = ...) -> Untyped: ...
def group_delay(system: Untyped, w: int = 512, whole: bool = False, fs: Untyped = ...) -> Untyped: ...
def sosfreqz(sos: Untyped, worN: int = 512, whole: bool = False, fs: Untyped = ...) -> Untyped: ...
def tf2zpk(b: Untyped, a: Untyped) -> Untyped: ...
def zpk2tf(z: Untyped, p: Untyped, k: Untyped) -> Untyped: ...
def tf2sos(b: Untyped, a: Untyped, pairing: Untyped | None = None, *, analog: bool = False) -> Untyped: ...
def sos2tf(sos: Untyped) -> Untyped: ...
def sos2zpk(sos: Untyped) -> Untyped: ...
def zpk2sos(z: Untyped, p: Untyped, k: Untyped, pairing: Untyped | None = None, *, analog: bool = False) -> Untyped: ...
def normalize(b: Untyped, a: Untyped) -> Untyped: ...
def lp2lp(b: Untyped, a: Untyped, wo: float = 1.0) -> Untyped: ...
def lp2hp(b: Untyped, a: Untyped, wo: float = 1.0) -> Untyped: ...
def lp2bp(b: Untyped, a: Untyped, wo: float = 1.0, bw: float = 1.0) -> Untyped: ...
def lp2bs(b: Untyped, a: Untyped, wo: float = 1.0, bw: float = 1.0) -> Untyped: ...
def bilinear(b: Untyped, a: Untyped, fs: float = 1.0) -> Untyped: ...
def iirdesign(
    wp: Untyped,
    ws: Untyped,
    gpass: Untyped,
    gstop: Untyped,
    analog: bool = False,
    ftype: str = "ellip",
    output: str = "ba",
    fs: Untyped | None = None,
) -> Untyped: ...
def iirfilter(
    N: Untyped,
    Wn: Untyped,
    rp: Untyped | None = None,
    rs: Untyped | None = None,
    btype: str = "band",
    analog: bool = False,
    ftype: str = "butter",
    output: str = "ba",
    fs: Untyped | None = None,
) -> Untyped: ...
def bilinear_zpk(z: Untyped, p: Untyped, k: Untyped, fs: Untyped) -> Untyped: ...
def lp2lp_zpk(z: Untyped, p: Untyped, k: Untyped, wo: float = 1.0) -> Untyped: ...
def lp2hp_zpk(z: Untyped, p: Untyped, k: Untyped, wo: float = 1.0) -> Untyped: ...
def lp2bp_zpk(z: Untyped, p: Untyped, k: Untyped, wo: float = 1.0, bw: float = 1.0) -> Untyped: ...
def lp2bs_zpk(z: Untyped, p: Untyped, k: Untyped, wo: float = 1.0, bw: float = 1.0) -> Untyped: ...
def butter(
    N: Untyped,
    Wn: Untyped,
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    fs: Untyped | None = None,
) -> Untyped: ...
def cheby1(
    N: Untyped,
    rp: Untyped,
    Wn: Untyped,
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    fs: Untyped | None = None,
) -> Untyped: ...
def cheby2(
    N: Untyped,
    rs: Untyped,
    Wn: Untyped,
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    fs: Untyped | None = None,
) -> Untyped: ...
def ellip(
    N: Untyped,
    rp: Untyped,
    rs: Untyped,
    Wn: Untyped,
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    fs: Untyped | None = None,
) -> Untyped: ...
def bessel(
    N: Untyped,
    Wn: Untyped,
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    norm: str = "phase",
    fs: Untyped | None = None,
) -> Untyped: ...
def maxflat() -> None: ...
def yulewalk() -> None: ...
def band_stop_obj(
    wp: Untyped,
    ind: Untyped,
    passb: Untyped,
    stopb: Untyped,
    gpass: Untyped,
    gstop: Untyped,
    type: Untyped,
) -> Untyped: ...
def buttord(
    wp: Untyped,
    ws: Untyped,
    gpass: Untyped,
    gstop: Untyped,
    analog: bool = False,
    fs: Untyped | None = None,
) -> Untyped: ...
def cheb1ord(
    wp: Untyped,
    ws: Untyped,
    gpass: Untyped,
    gstop: Untyped,
    analog: bool = False,
    fs: Untyped | None = None,
) -> Untyped: ...
def cheb2ord(
    wp: Untyped,
    ws: Untyped,
    gpass: Untyped,
    gstop: Untyped,
    analog: bool = False,
    fs: Untyped | None = None,
) -> Untyped: ...
def ellipord(
    wp: Untyped,
    ws: Untyped,
    gpass: Untyped,
    gstop: Untyped,
    analog: bool = False,
    fs: Untyped | None = None,
) -> Untyped: ...
def buttap(N: Untyped) -> Untyped: ...
def cheb1ap(N: Untyped, rp: Untyped) -> Untyped: ...
def cheb2ap(N: Untyped, rs: Untyped) -> Untyped: ...
def ellipap(N: Untyped, rp: Untyped, rs: Untyped) -> Untyped: ...
def besselap(N: Untyped, norm: str = "phase") -> Untyped: ...
def iirnotch(w0: Untyped, Q: Untyped, fs: float = 2.0) -> Untyped: ...
def iirpeak(w0: Untyped, Q: Untyped, fs: float = 2.0) -> Untyped: ...
def iircomb(w0: Untyped, Q: Untyped, ftype: str = "notch", fs: float = 2.0, *, pass_zero: bool = False) -> Untyped: ...
def gammatone(
    freq: Untyped,
    ftype: Untyped,
    order: Untyped | None = None,
    numtaps: Untyped | None = None,
    fs: Untyped | None = None,
) -> Untyped: ...
