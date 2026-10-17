# ----------------------- South African WFA scale -----------------------
# Official chart values are in pounds. Race Edge intentionally uses the
# simplified conversion requested by the analyst: 1 lb = 0.5 kg. Since
# 1 kg = 2 MR points, 1 lb of WFA is therefore exactly 1 MR point.
_WFA_MONTHS = [
    "August", "September", "October", "November", "December", "January",
    "February", "March", "April", "May", "June", "July",
]

_WFA_LB = {
    "LE1200": {
        2: [0, 0, 0, 0, 0, 0, 0, 0, 21, 19, 17, 15],
        3: [14, 13, 11, 10, 8, 7, 6, 5, 4, 3, 2, 1],
        4: [0] * 12,
    },
    "1201_1400": {
        2: [0, 0, 0, 0, 0, 0, 0, 0, 24, 22, 20, 18],
        3: [17, 16, 14, 12, 10, 9, 7, 6, 5, 4, 3, 2],
        4: [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "1401_1600": {
        2: [0, 0, 0, 0, 0, 0, 0, 0, 25, 23, 21, 19],
        3: [18, 17, 16, 14, 12, 10, 8, 6, 5, 4, 3, 2],
        4: [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "1601_2000": {
        2: [0, 0, 0, 0, 0, 0, 0, 0, 27, 25, 23, 21],
        3: [20, 19, 18, 16, 14, 12, 10, 9, 7, 5, 4, 3],
        4: [2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    },
    "2001_2400": {
        2: [0] * 12,
        3: [21, 20, 19, 17, 16, 14, 12, 10, 9, 7, 6, 4],
        4: [3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0],
    },
    "2401_3600": {
        2: [0] * 12,
        3: [23, 22, 21, 19, 18, 16, 14, 12, 11, 9, 8, 7],
        4: [4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1],
    },
}

_WFA_BAND_LABELS = {
    "LE1200": "≤1200 m",
    "1201_1400": "1201–1400 m",
    "1401_1600": "1401–1600 m",
    "1601_2000": "1601–2000 m",
    "2001_2400": "2001–2400 m",
    "2401_3600": "2401–3600 m",
}

def wfa_distance_band(distance_m: float) -> str:
    d = float(distance_m)
    if d <= 1200:
        return "LE1200"
    if d <= 1400:
        return "1201_1400"
    if d <= 1600:
        return "1401_1600"
    if d <= 2000:
        return "1601_2000"
    if d <= 2400:
        return "2001_2400"
    if d <= 3600:
        return "2401_3600"
    raise ValueError("The built-in WFA chart only covers races up to 3600 m.")

def get_wfa_lb(race_date, distance_m: float, age: int) -> float:
    band = wfa_distance_band(distance_m)
    month_name = race_date.strftime("%B")
    month_idx = _WFA_MONTHS.index(month_name)
    age = int(age)
    if age >= 5:
        return 0.0
    if age not in (2, 3, 4):
        raise ValueError(f"Unsupported horse age: {age}")
    return float(_WFA_LB[band][age][month_idx])

