"""Central config: paths, constants, 2026 World Cup setup."""
from pathlib import Path

ROOT = Path(__file__).parent
DB_PATH = ROOT / "db" / "wcp.sqlite"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

RESULTS_CSV_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

TRAIN_CUTOFF = "2025-01-01"
TEST_CUTOFF = "2025-12-31"
MIN_TRAIN_YEAR = 2018

ELO_K = 30.0
ELO_INITIAL = 1500.0
ELO_HOME_ADV = 65.0

QSVM_PCA_COMPONENTS = 6
QSVM_FEATURE_MAP_REPS = 2
QAOA_REPS = 2

# 2026 FIFA World Cup — 48 teams, 12 groups of 4.
# Top 2 of each group + 8 best third-place teams advance to round of 32.
# Source: official FIFA Final Draw, 5 Dec 2025 (Washington DC).
WC_2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

WC_2026_HOSTS = {"United States", "Mexico", "Canada"}


# ---------------------------------------------------------------------------
# 2026 FIFA World Cup knockout bracket — fixed match graph.
# Source: FIFA Competition Regulations 2026, knockout-stage appendix.
# Each Round of 32 slot is either a fixed group placement ("1A", "2A") or
# a wildcard third-place slot ("3:ABCDF") meaning "the third-placed team
# from one of A/B/C/D/F", resolved at run time by the qualified-third-place
# assignment table.
#
# Match numbering follows the official schedule:
#   Round of 32:  Matches 73–88   (16)
#   Round of 16:  Matches 89–96   (8)
#   Quarter-final: Matches 97–100 (4)
#   Semi-final:    Matches 101–102 (2)
#   Final:         Match 104       (Match 103 is third-place play-off, not modelled)
# ---------------------------------------------------------------------------

WC_2026_R32 = [
    # (match_number, slot_top, slot_bottom)
    (73, "2A", "2B"),
    (74, "1E", "3:ABCDF"),
    (75, "1F", "2C"),
    (76, "1C", "2F"),
    (77, "1I", "3:CDFGH"),
    (78, "2E", "2I"),
    (79, "1A", "3:CEFHI"),
    (80, "1L", "3:EHIJK"),
    (81, "1D", "3:BEFIJ"),
    (82, "1G", "3:AEHIJ"),
    (83, "2K", "2L"),
    (84, "1H", "2J"),
    (85, "1B", "3:EFGIJ"),
    (86, "1J", "2H"),
    (87, "1K", "3:DEIJL"),
    (88, "2D", "2G"),
]

# Round of 16: (this_match, winner_of_match_A, winner_of_match_B)
WC_2026_R16 = [
    (89, 74, 77),
    (90, 73, 75),
    (91, 76, 78),
    (92, 79, 80),
    (93, 83, 84),
    (94, 81, 82),
    (95, 86, 88),
    (96, 85, 87),
]

WC_2026_QF = [
    (97, 89, 90),
    (98, 93, 94),
    (99, 91, 92),
    (100, 95, 96),
]

WC_2026_SF = [
    (101, 97, 98),
    (102, 99, 100),
]

WC_2026_FINAL = (104, 101, 102)
