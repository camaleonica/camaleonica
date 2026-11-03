#!/usr/bin/env python3
"""Generate the animated GitHub profile banners (logos only, no portrait).

Run from the repository root:
    python scripts/banner/generate.py

The left panel morphs between three silhouettes (code, gear, star) using
optimal-transport point travellers. Edit ROWS to change the right panel.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
LOGOS = Path(__file__).resolve().parent / "logos"
DATA = Path(__file__).resolve().parent / "data"

W, H = 1180, 610
HOLD = 2.4          # seconds each logo stays still
MORPH = 1.3         # seconds each transition takes
TRAVELLER_COUNT = 1600
SEED = 314159
HANDLE = "@camaleonica"

ROWS = [
    ("Subject", "Candela"),
    ("Role", "Computer Engineering Student"),
    ("Origin", "Buenos Aires · Argentina"),
    ("Education", "UADE · Ingeniería Informática"),
    ("Status", "Building + Learning + Shipping"),
    ("ToolChain", "VS Code · Eclipse · Git"),
    ("Core.Lang", "Java · Python · TypeScript"),
    ("Core.Frontend", "React · HTML · CSS"),
    ("Core.Backend", "Spring · Node"),
    ("Core.Database", "MySQL · MongoDB · Neo4j"),
    ("Core.Infra", "Docker · Linux"),
    ("Grid.Mail", "ricocandel4@gmail.com"),
    ("Grid.LinkedIn", "/in/candela-celeste-rico"),
    ("Grid.GitHub", "camaleonica"),
    ("Grid.Portfolio", "portfolioceleste.vercel.app"),
]

THEMES = {
    "dark": {
        "bg": "#0A101F", "panel": "#0D1628", "panel2": "#101B30", "line": "#25344C",
        "muted": "#8291A8", "text": "#DDE7F5", "dots": "#AA9BEF", "chrome": "#22D3EE",
        "accent": "#10B981", "shadow": "#02050B",
    },
    "light": {
        "bg": "#F6F8FA", "panel": "#FFFFFF", "panel2": "#EDF3F7", "line": "#CBD7E1",
        "muted": "#64748B", "text": "#172033", "dots": "#4A3D7A", "chrome": "#0891B2",
        "accent": "#10B981", "shadow": "#AAB7C4",
    },
}

MONO = "ui-monospace,SFMono-Regular,Consolas,monospace"


def make_logos() -> dict[str, Image.Image]:
    """Create clean 400px black-on-transparent silhouette sources."""
    LOGOS.mkdir(parents=True, exist_ok=True)
    size = 400
    logos: dict[str, Image.Image] = {}

    # Rust-inspired gear: twelve teeth, heavy annulus, and hub cutout.
    rust = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(rust)
    center = np.array([200.0, 200.0])
    outer: list[tuple[float, float]] = []
    for tooth in range(12):
        base = tooth * math.tau / 12
        for offset, radius in [
            (-0.42, 150), (-0.30, 178), (0.30, 178), (0.42, 150)
        ]:
            a = base + offset * math.tau / 12
            outer.append(tuple(center + radius * np.array([math.cos(a), math.sin(a)])))
    d.polygon(outer, fill="black")
    d.ellipse((70, 70, 330, 330), fill="black")
    d.ellipse((128, 128, 272, 272), fill=(0, 0, 0, 0))
    d.ellipse((174, 174, 226, 226), fill="black")
    logos["rust"] = rust

    # </> mark built from broad, rounded strokes.
    code = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(code)
    stroke = 42
    d.line([(154, 95), (66, 200), (154, 305)], fill="black", width=stroke, joint="curve")
    d.line([(246, 95), (334, 200), (246, 305)], fill="black", width=stroke, joint="curve")
    d.line([(225, 72), (174, 328)], fill="black", width=stroke)
    logos["code"] = code

    # Stellar-inspired four-point star with a small connected node network.
    stellar = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(stellar)
    star: list[tuple[float, float]] = []
    for i in range(16):
        a = -math.pi / 2 + i * math.pi / 8
        radius = 137 if i % 4 == 0 else (45 if i % 2 == 0 else 25)
        star.append((200 + math.cos(a) * radius, 200 + math.sin(a) * radius))
    d.polygon(star, fill="black")
    nodes = [(82, 112), (322, 106), (326, 300), (88, 316)]
    for a, b in zip(nodes, nodes[1:] + nodes[:1]):
        d.line([a, b], fill="black", width=12)
    for x, y in nodes:
        d.ellipse((x - 19, y - 19, x + 19, y + 19), fill="black")
    logos["stellar"] = stellar

    for name, image in logos.items():
        image.save(LOGOS / f"{name}.png", optimize=True)
    return logos


def sample_logo_points(
    image: Image.Image, rng: np.random.Generator, count: int
) -> np.ndarray:
    """Sample a silhouette into the portrait frame's visual coordinate space."""
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.where(alpha > 127)
    chosen = rng.choice(len(xs), count, replace=len(xs) < count)
    # Logo occupies a centered 270x270 square inside VISUAL.MAP.
    return np.column_stack((89 + xs[chosen] * 0.675, 188 + ys[chosen] * 0.675)).astype(
        np.float32
    )


def transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Order target points by minimum-cost assignment from source points."""
    rows, cols = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
    ordered = np.empty_like(target)
    ordered[rows] = target[cols]
    return ordered


def num(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def point_path(points: np.ndarray) -> str:
    """Aggregate adjacent horizontal one-pixel dots into compact SVG path runs."""
    if not len(points):
        return ""
    integer = np.rint(points).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in integer}, key=lambda p: (p[1], p[0]))
    chunks: list[str] = []
    i = 0
    while i < len(unique):
        x0, y = unique[i]
        x1 = x0
        i += 1
        while i < len(unique) and unique[i][1] == y and unique[i][0] <= x1 + 1:
            x1 = unique[i][0]
            i += 1
        chunks.append(f"M{x0} {y}h{x1 - x0 + 1}")
    return "".join(chunks)


def dotted_leader(x1: float, x2: float, y: float) -> str:
    if x2 <= x1:
        return ""
    return "".join(f"M{x} {num(y)}h1" for x in np.arange(x1, x2, 5.0))


def text_width(text: str, font_size: float) -> float:
    """Stable monospace width used both for textLength and leader placement."""
    return len(text) * font_size * 0.605



def render_svg(theme_name: str, pts: dict[str, np.ndarray], rng: np.random.Generator) -> str:
    t = THEMES[theme_name]
    code = pts["code"]
    gear = transport(code, pts["rust"])
    star = transport(gear, pts["stellar"])
    # Each traveller returns to its own starting point, so the loop is seamless.
    frames = [code, code, gear, gear, star, star, code]
    times = [0.0]
    for _ in range(3):
        times += [times[-1] + HOLD, times[-1] + HOLD + MORPH]
    times = times[:7]
    loop = times[-1]
    key_times = ";".join(num(v / loop) for v in times)
    splines = ";".join(["0 0 1 1", ".45 0 .2 1"] * 3)
    n = len(code)

    parts: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
        'aria-labelledby="title desc">',
        "<title id=\"title\">Candela's live system profile</title>",
        '<desc id="desc">Animated terminal profile with morphing code, gear and star silhouettes.</desc>',
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
        f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" '
        'flood-opacity=".28"/></filter>',
        '<filter id="glow" x="-100%" y="-100%" width="300%" height="300%">'
        f'<feGaussianBlur stdDeviation="3" result="b"/><feFlood flood-color="{t["chrome"]}" '
        'flood-opacity=".35"/><feComposite in2="b" operator="in"/>'
        '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<clipPath id="visualClip"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>',
        "</defs>",
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" '
        f'stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/>'
        '<circle cx="59" cy="38" r="6" fill="#FEBC2E"/>'
        '<circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" font-family="{MONO}" '
        'font-size="13" letter-spacing=".4">profile.sh --live</text>',
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" font-family="{MONO}" font-size="13" '
        'font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" font-family="{MONO}" '
        'font-size="11">270×270 / 1-BIT</text>',
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12'
        f'M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        f'<g clip-path="url(#visualClip)" shape-rendering="crispEdges" fill="{t["dots"]}">',
    ]

    # Each traveller sits on the code mark by default (so static renderers still
    # show a logo), fades in on a staggered schedule, then loops through the logos.
    fade_start = rng.uniform(0.0, 1.1, size=n)
    intro = 1.6
    for i in range(n):
        x0, y0 = code[i]
        positions = ";".join(f"{num(f[i, 0])} {num(f[i, 1])}" for f in frames)
        k = num(fade_start[i] / (fade_start[i] + 0.5))
        parts.append(
            f'<path transform="translate({num(x0)} {num(y0)})" d="M-.75-.75h1.5v1.5h-1.5z">'
            f'<animate attributeName="opacity" dur="{num(fade_start[i] + 0.5)}s" '
            f'keyTimes="0;{k};1" values="0;0;1" fill="freeze"/>'
            f'<animateTransform attributeName="transform" type="translate" begin="{intro}s" '
            f'dur="{num(loop)}s" repeatCount="indefinite" calcMode="spline" '
            f'keyTimes="{key_times}" keySplines="{splines}" values="{positions}"/></path>'
        )
    parts.append("</g>")

    parts.extend([
        f'<text x="58" y="551" fill="{t["muted"]}" font-family="{MONO}" font-size="10">'
        f'PTS {n:05d} · OPTIMAL TRANSPORT</text>',
        f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
        f'<text x="490" y="111" fill="{t["chrome"]}" font-family="{MONO}" font-size="13" '
        'font-weight="700" letter-spacing="1.2">SYSTEM.INFO</text>',
        '<g filter="url(#glow)"><circle cx="895" cy="106" r="4" fill="#FF4D5A">'
        '<animate attributeName="opacity" values="1;.3;1" dur="1.6s" repeatCount="indefinite"/>'
        '</circle></g>',
        f'<text x="907" y="111" fill="#FF4D5A" font-family="{MONO}" font-size="12" '
        'font-weight="700">LIVE</text>',
        f'<rect x="962" y="94" width="166" height="24" rx="12" fill="{t["chrome"]}" opacity=".16" '
        f'stroke="{t["chrome"]}"/>',
        f'<text x="1045" y="111" text-anchor="middle" fill="{t["chrome"]}" font-family="{MONO}" '
        f'font-size="14" font-weight="700">{html.escape(HANDLE)}</text>',
    ])

    value_right = 1127.0
    row_y = 153.0
    for label, value in ROWS:
        value_len = text_width(value, 14)
        label_len = text_width(label, 14)
        leader_start = 491 + label_len + 12
        leader_end = value_right - value_len - 12
        parts.extend([
            f'<text x="491" y="{num(row_y)}" fill="{t["muted"]}" font-family="{MONO}" '
            f'font-size="14">{html.escape(label)}</text>',
            f'<path d="{dotted_leader(leader_start, leader_end, row_y - 4)}" fill="none" '
            f'stroke="{t["line"]}" stroke-width="1" shape-rendering="crispEdges"/>',
            f'<text x="{num(value_right)}" y="{num(row_y)}" text-anchor="end" fill="{t["text"]}" '
            f'font-family="{MONO}" font-size="14" textLength="{num(value_len)}" '
            f'lengthAdjust="spacingAndGlyphs">{html.escape(value)}</text>',
        ])
        row_y += 23

    parts.extend([
        f'<path d="M490 530H1130" stroke="{t["line"]}"/>',
        f'<text x="491" y="548" fill="{t["accent"]}" font-family="{MONO}" font-size="11">'
        "● ALL SYSTEMS NOMINAL</text>",
        f'<text x="1128" y="548" text-anchor="end" fill="{t["muted"]}" font-family="{MONO}" '
        'font-size="11">UTC-3 · LATAM NODE</text>',
        "</svg>",
    ])
    return "".join(parts)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    logos = make_logos()
    for index, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + 100 + index)
        sampled = {name: sample_logo_points(img, rng, TRAVELLER_COUNT) for name, img in logos.items()}
        for name, points in sampled.items():
            np.save(DATA / f"{name}-{theme}.npy", points)
        svg = render_svg(theme, sampled, rng)
        output = ASSETS / f"banner-{theme}.svg"
        output.write_text(svg, encoding="utf-8")
        size = output.stat().st_size
        print(f"{output.relative_to(ROOT)}: {size:,} bytes ({size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
