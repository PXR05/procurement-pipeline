import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("BD_PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent)
).resolve()
DATA_DIR = Path(os.environ.get("BD_DATA_DIR", PROJECT_ROOT / "data")).resolve()

RAW_DIR = DATA_DIR / "raw"
BRONZE_OPENTENDER_DIR = RAW_DIR / "opentender"
BRONZE_DJPK_DIR = RAW_DIR / "djpk"
BRONZE_BPK_DIR = RAW_DIR / "bpk"
BRONZE_LKPP_DIR = RAW_DIR / "lkpp"
MANIFESTS_DIR = DATA_DIR / "manifests"
QUARANTINE_DIR = DATA_DIR / "quarantine"

SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
OUTPUT_DIR = Path(os.environ.get("BD_OUTPUT_DIR", PROJECT_ROOT / "output")).resolve()
CHARTS_DIR = OUTPUT_DIR / "charts"


MIN_VALID_YEAR = 2008
MAX_VALID_YEAR = 2026
VALID_CURRENCY = "IDR"


IT_KEYWORDS = {
    "hardware": [
        "komputer",
        "laptop",
        "pc",
        "server",
        "storage",
        "workstation",
        "scanner",
        "printer",
        "ups",
        "switch",
        "router",
        "firewall",
        "access point",
        "cctv",
        "videowall",
        "display",
        "tablet",
        "blade server",
        "nas",
        "san",
        "harddisk",
        "ssd",
        "ram",
        "processor",
    ],
    "software": [
        "software",
        "perangkat lunak",
        "aplikasi",
        "sistem informasi",
        "lisensi",
        "license",
        "portal",
        "platform",
        "website",
        "erp",
        "database",
        "crm",
        "cms",
        "antivirus",
        "operating system",
        "os",
        "middleware",
        "api",
        "mobile app",
    ],
    "infrastructure_network": [
        "jaringan komputer",
        "jaringan internet",
        "fiber optic",
        "fo",
        "bandwidth",
        "lan",
        "wan",
        "vpn",
        "data center",
        "pusat data",
        "colocation",
        "cloud",
        "hosting",
        "intranet",
        "kabel utp",
        "rack server",
    ],
    "services_security": [
        "keamanan siber",
        "cyber security",
        "pemeliharaan ti",
        "maintenance ti",
        "pemeliharaan sistem",
        "konsultansi ti",
        "helpdesk",
        "instalasi jaringan",
        "penetration test",
        "pentest",
        "audit ti",
        "soc",
        "siem",
        "managed service",
        "pengembangan sistem",
        "integrasi sistem",
    ],
}


IT_EXCLUSION_KEYWORDS = [
    "jaringan irigasi",
    "jaringan air",
    "jaringan pipa",
    "jaringan gas",
    "jaringan listrik",
    "jaringan jalan",
    "server makanan",
    "server minuman",
    "komputer kasir restoran",
    "alat kesehatan",
    "medis",
    "bangunan",
    "gedung",
    "konstruksi jalan",
    "drainase",
]


HHI_HIGH_CONCENTRATION_THRESHOLD = 0.25
HHI_MODERATE_CONCENTRATION_THRESHOLD = 0.15
SUPPLIER_TOP1_SHARE_ALERT = 0.50
AWARD_ESTIMATE_RATIO_NEAR_CEILING = 0.995
AWARD_ESTIMATE_RATIO_ANOMALOUS_LOW = 0.60
MIN_AWARDS_FOR_CONCENTRATION_EVAL = 3
