"""Select the currently reachable SIMBAD and VizieR mirrors."""

from functools import lru_cache
from urllib.parse import urlencode
from urllib.request import urlopen

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier


SIMBAD_SERVERS = (
    "simbad.cds.unistra.fr",
    "simbad.cfa.harvard.edu",
    "simbad.idia.ac.za",
)

VIZIER_MIRRORS = (
    ("https", "vizier.cds.unistra.fr"),
    ("https", "vizier.idia.ac.za"),
    ("http", "vizier.nao.ac.jp"),
    ("https", "vizier.cfa.harvard.edu"),
)


@lru_cache(maxsize=1)
def available_simbad_server(timeout=15):
    """Return the first responsive validated SIMBAD TAP mirror."""
    for index, server in enumerate(SIMBAD_SERVERS):
        server_timeout = min(4, timeout) if index == 0 else timeout
        try:
            with urlopen(
                f"https://{server}/simbad/sim-tap/capabilities",
                timeout=server_timeout,
            ):
                return server
        except Exception:
            continue
    raise ConnectionError("No SIMBAD mirror is currently reachable.")


@lru_cache(maxsize=1)
def available_vizier_mirror(timeout=15):
    """Return the first VizieR mirror that completes a small catalogue query."""
    parameters = urlencode({
        "-source": "I/239/hip_main",
        "HIP": "1",
        "-out.max": "1",
    })
    for index, (scheme, server) in enumerate(VIZIER_MIRRORS):
        server_timeout = min(4, timeout) if index < 2 else timeout
        url = f"{scheme}://{server}/viz-bin/asu-tsv?{parameters}"
        try:
            with urlopen(url, timeout=server_timeout) as response:
                sample = response.read(512)
            if b"VizieR Astronomical Server" in sample:
                return scheme, server
        except Exception:
            continue
    raise ConnectionError("No VizieR mirror is currently reachable.")


def configured_simbad(timeout=20, server=None):
    """Build a SIMBAD client using the first responsive validated mirror."""
    simbad = Simbad()
    # astroquery 0.4.11 validates ``server`` against an outdated fixed list.
    # Setting the backing value preserves its normal TAP client while allowing
    # the current IDIA and CfA mirror domains.
    simbad._server = server or available_simbad_server()
    simbad.timeout = timeout
    return simbad


def _configured_vizier_for(scheme, server, **kwargs):
    """Build one VizieR client for an explicitly selected mirror."""
    kwargs.setdefault("timeout", 15)
    vizier = Vizier(vizier_server=server, **kwargs)
    if scheme == "http":
        original_url_builder = vizier._server_to_url

        def mirror_url(return_type="votable"):
            https_url = original_url_builder(return_type)
            return "http://" + https_url.removeprefix("https://")

        vizier._server_to_url = mirror_url

    original_request = vizier._request

    def mirror_request(method, url, **request_kwargs):
        # The RUNA catalogue requests are small. GET avoids mirrors that accept
        # the request but leave equivalent POST requests waiting indefinitely.
        if method == "POST":
            payload = request_kwargs.pop("data", None)
            if isinstance(payload, str):
                payload = [
                    line.split("=", 1)
                    for line in payload.splitlines()
                    if "=" in line
                ]
            request_kwargs["params"] = payload
            method = "GET"
        return original_request(method, url, **request_kwargs)

    vizier._request = mirror_request
    return vizier


def configured_vizier(**kwargs):
    """Build a VizieR client using the first responsive validated mirror."""
    return _configured_vizier_for(*available_vizier_mirror(), **kwargs)


def configured_viziers(**kwargs):
    """Yield clients for every validated VizieR mirror in fallback order."""
    for scheme, server in VIZIER_MIRRORS:
        yield _configured_vizier_for(scheme, server, **kwargs)
