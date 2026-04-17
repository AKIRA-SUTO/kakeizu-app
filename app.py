
from pathlib import Path
from io import BytesIO, StringIO
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
from flask import Flask, request, send_file

app = Flask(__name__)

Y_BY_GEN = {-5: 5.2, -4: 4.4, -3: 3.6, -2: 2.75, -1: 1.9, 0: 0.95, 1: -0.2, 2: -1.2}
UNIT = 1.0

def pick_japanese_font():
    from pathlib import Path

    # ←ここが今回の本命
    bundled = Path("fonts/EnokoroSans-vert-Normal.otf")
    if bundled.exists():
        return str(bundled)

    # 念のための保険
    candidates = [
        "Noto Sans CJK JP",
        "IPAexGothic",
        "Yu Gothic",
        "MS Gothic",
    ]
    installed = {f.name: f.fname for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return installed[name]

    return fm.findfont("DejaVu Sans")

FONT_PROP = fm.FontProperties(fname=pick_japanese_font())

def normalize_bool(v):
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"true", "1", "yes", "y"}

def normalize_str(v):
    if pd.isna(v):
        return ""
    return str(v).strip()

def read_persons_csv_text(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(csv_text))
    for col in ["person_id","display_name","sex","father_id","mother_id","spouse_group_id",
                "relative_type","branch_side"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_str)
    for col in ["generation","birth_order"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_display"] = df["is_display"].apply(normalize_bool) if "is_display" in df.columns else True
    df["is_bloodline"] = df["is_bloodline"].apply(normalize_bool) if "is_bloodline" in df.columns else True
    return df[df["person_id"] != ""].copy()

def validate_persons(persons: pd.DataFrame):
    errors = []
    required = [
        "person_id","display_name","sex","father_id","mother_id","spouse_group_id",
        "generation","relative_type","branch_side","birth_order","is_bloodline","is_display"
    ]
    for c in required:
        if c not in persons.columns:
            errors.append(f"列が不足しています: {c}")
    if errors:
        return errors
    if persons.empty:
        errors.append("CSVが空です。")
        return errors
    if not (persons["relative_type"] == "self").any():
        errors.append("relative_type='self' の人物が必要です。")
    if persons["person_id"].duplicated().any():
        errors.append("person_id が重複しています。")
    ids = set(persons["person_id"])
    for col in ["father_id", "mother_id"]:
        for val in persons[col]:
            v = normalize_str(val)
            if v and v not in ids:
                errors.append(f"{col} に未登録のIDがあります: {v}")
    return list(dict.fromkeys(errors))

def get_spouse(persons: pd.DataFrame, idx: dict, person_id: str):
    if person_id not in idx:
        return None
    sg = normalize_str(idx[person_id].spouse_group_id)
    if not sg:
        return None
    matches = [r.person_id for r in persons.itertuples(index=False)
               if normalize_str(r.spouse_group_id) == sg and r.person_id != person_id]
    return matches[0] if matches else None

def children_of(persons: pd.DataFrame, parent_id: str):
    rows = []
    for r in persons.itertuples(index=False):
        if normalize_str(r.father_id) == parent_id or normalize_str(r.mother_id) == parent_id:
            rows.append(r)
    return rows

def sort_children(rows):
    return sorted(rows, key=lambda r: (9999 if pd.isna(r.birth_order) else r.birth_order, r.person_id))

def person_width_units(persons: pd.DataFrame, idx: dict, person_id: str, visited=None):
    if visited is None:
        visited = set()
    if person_id in visited:
        return 1.0
    visited = set(visited)
    visited.add(person_id)
    kids = sort_children(children_of(persons, person_id))
    if not kids:
        return 1.0
    return max(1.0, sum(person_width_units(persons, idx, k.person_id, visited) for k in kids))

def ancestor_width_units(persons: pd.DataFrame, idx: dict, person_id: str, visited=None):
    if visited is None:
        visited = set()
    if person_id in visited:
        return 1.0
    visited = set(visited)
    visited.add(person_id)
    row = idx[person_id]
    father = normalize_str(row.father_id)
    mother = normalize_str(row.mother_id)
    widths = []
    if father and father in idx:
        widths.append(ancestor_width_units(persons, idx, father, visited))
    if mother and mother in idx:
        widths.append(ancestor_width_units(persons, idx, mother, visited))
    return max(1.0, sum(widths) if widths else 1.0)

def layout_descendants(persons: pd.DataFrame, idx: dict, center_person_id: str, pos: dict):
    self_row = idx[center_person_id]
    spouse_id = get_spouse(persons, idx, center_person_id)
    if spouse_id:
        if self_row.sex == "male":
            pos[spouse_id] = (-UNIT, Y_BY_GEN[0])
        elif self_row.sex == "female":
            pos[spouse_id] = (UNIT, Y_BY_GEN[0])
        else:
            pos[spouse_id] = (-UNIT, Y_BY_GEN[0])

    kids = sort_children(children_of(persons, center_person_id))
    if not kids:
        return

    widths = {k.person_id: person_width_units(persons, idx, k.person_id) for k in kids}
    total_width = sum(widths.values())
    x_left = -total_width / 2.0

    child_centers = {}
    cursor = x_left
    for kid in kids:
        w = widths[kid.person_id]
        center = cursor + w / 2.0
        child_centers[kid.person_id] = center
        cursor += w

    for kid in kids:
        x = -child_centers[kid.person_id]
        pos[kid.person_id] = (x, Y_BY_GEN.get(1, -0.2))

def layout_siblings(persons: pd.DataFrame, idx: dict, self_id: str, pos: dict):
    self_row = idx[self_id]
    father_id = normalize_str(self_row.father_id)
    mother_id = normalize_str(self_row.mother_id)
    siblings = []
    for r in persons.itertuples(index=False):
        if r.person_id == self_id:
            continue
        if int(r.generation) == 0 and normalize_str(r.father_id) == father_id and normalize_str(r.mother_id) == mother_id:
            siblings.append(r)

    older = [r for r in siblings if r.relative_type in ("older_brother","older_sister")]
    younger = [r for r in siblings if r.relative_type in ("younger_brother","younger_sister")]
    older = sorted(older, key=lambda r: (9999 if pd.isna(r.birth_order) else r.birth_order, r.person_id))
    younger = sorted(younger, key=lambda r: (9999 if pd.isna(r.birth_order) else r.birth_order, r.person_id))

    for i, r in enumerate(older, start=1):
        pos[r.person_id] = (i * 1.15, Y_BY_GEN[0])
    for i, r in enumerate(younger, start=1):
        pos[r.person_id] = (-i * 1.15, Y_BY_GEN[0])

def layout_ancestors(persons: pd.DataFrame, idx: dict, root_person_id: str, pos: dict):
    root = idx[root_person_id]
    father_id = normalize_str(root.father_id)
    mother_id = normalize_str(root.mother_id)

    if father_id and father_id in idx:
        fw = ancestor_width_units(persons, idx, father_id)
        pos[father_id] = (max(2.4, fw * 0.6), Y_BY_GEN[-1])
    if mother_id and mother_id in idx:
        mw = ancestor_width_units(persons, idx, mother_id)
        pos[mother_id] = (-max(2.4, mw * 0.6), Y_BY_GEN[-1])

    def place_lineage(person_id: str, side_sign: int):
        if person_id not in idx or person_id not in pos:
            return
        row = idx[person_id]
        gen = int(row.generation) if not pd.isna(row.generation) else 0
        father = normalize_str(row.father_id)
        mother = normalize_str(row.mother_id)
        base_x, _ = pos[person_id]
        target_y = Y_BY_GEN.get(gen - 1, min(Y_BY_GEN.values()) - 0.8)

        f_width = ancestor_width_units(persons, idx, father) if father and father in idx else 1.0
        m_width = ancestor_width_units(persons, idx, mother) if mother and mother in idx else 1.0
        gap = 0.55

        if father and father in idx and mother and mother in idx:
            father_center = base_x + side_sign * ((m_width + gap) / 2.0)
            mother_center = base_x - side_sign * ((f_width + gap) / 2.0)
            pos[father] = (father_center, target_y)
            pos[mother] = (mother_center, target_y)
        else:
            if father and father in idx:
                pos[father] = (base_x + side_sign * max(0.7, f_width / 2.0), target_y)
            if mother and mother in idx:
                pos[mother] = (base_x - side_sign * max(0.7, m_width / 2.0), target_y)

        if father and father in idx:
            place_lineage(father, side_sign)
        if mother and mother in idx:
            place_lineage(mother, side_sign)

    if father_id and father_id in idx:
        place_lineage(father_id, +1)
    if mother_id and mother_id in idx:
        place_lineage(mother_id, -1)

def build_layout(persons: pd.DataFrame):
    persons = persons[persons["is_display"] == True].copy()
    idx = {r.person_id: r for r in persons.itertuples(index=False)}
    self_rows = persons.loc[persons["relative_type"] == "self", "person_id"]
    if self_rows.empty:
        raise ValueError("relative_type='self' の人物が必要です。")
    self_id = self_rows.iloc[0]
    pos = {self_id: (0.0, Y_BY_GEN[0])}
    layout_siblings(persons, idx, self_id, pos)
    layout_descendants(persons, idx, self_id, pos)
    layout_ancestors(persons, idx, self_id, pos)
    return pos

def jp_vertical_text(name: str) -> str:
    s = normalize_str(name)
    if not s:
        return ""
    if " " in s:
        parts = [p for p in s.split(" ") if p]
    elif "　" in s:
        parts = [p for p in s.split("　") if p]
    else:
        parts = [s]
    if len(parts) >= 2:
        surname = "\n".join(list(parts[0]))
        given = "\n".join(list("".join(parts[1:])))
        return surname + "\n　\n" + given
    return "\n".join(list(s))

def box_size_for_name(name: str):
    lines = len(jp_vertical_text(name).split("\n"))
    width = 0.64
    height = max(1.55, 0.22 * lines + 0.58)
    return width, height

def create_family_tree(persons: pd.DataFrame) -> BytesIO:
    pos = build_layout(persons)
    sizes = {r.person_id: box_size_for_name(r.display_name) for r in persons.itertuples(index=False)}

    def top_of(pid):
        x, y = pos[pid]
        _, h = sizes[pid]
        return x, y + h / 2

    def bottom_of(pid):
        x, y = pos[pid]
        _, h = sizes[pid]
        return x, y - h / 2

    def draw_person_box(ax, x, y, label):
        bw, bh = box_size_for_name(label)
        rect = Rectangle((x - bw / 2, y - bh / 2), bw, bh, fill=False, linewidth=1.35)
        ax.add_patch(rect)
        ax.text(x, y, jp_vertical_text(label), ha="center", va="center",
                fontproperties=FONT_PROP, fontsize=12, linespacing=1.05)

    def hline(ax, x1, x2, y):
        ax.add_line(Line2D([x1, x2], [y, y], linewidth=1.05))

    def vline(ax, x, y1, y2):
        ax.add_line(Line2D([x, x], [y1, y2], linewidth=1.05))

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.axis("off")
    ax.set_xlim(min(xs) - 2.0, max(xs) + 2.0)
    ax.set_ylim(min(ys) - 1.3, max(ys) + 1.0)

    drawn_sg = set()
    for r in persons.itertuples(index=False):
        sg = normalize_str(r.spouse_group_id)
        if not sg or sg in drawn_sg:
            continue
        members = [x.person_id for x in persons.itertuples(index=False)
                   if normalize_str(x.spouse_group_id) == sg and x.person_id in pos]
        if len(members) == 2:
            p1, p2 = members
            x1, y1 = pos[p1]
            x2, y2 = pos[p2]
            if abs(y1 - y2) < 1e-6:
                hline(ax, min(x1, x2), max(x1, x2), y1)
        drawn_sg.add(sg)

    family_groups = {}
    for r in persons.itertuples(index=False):
        if r.person_id not in pos:
            continue
        f = normalize_str(r.father_id)
        m = normalize_str(r.mother_id)
        if f or m:
            family_groups.setdefault((f, m), []).append(r.person_id)

    for (f, m), child_ids in family_groups.items():
        child_ids = [c for c in child_ids if c in pos]
        if not child_ids:
            continue
        if f in pos and m in pos:
            fx, fy = pos[f]
            mx, my = pos[m]
            hline(ax, min(fx, mx), max(fx, mx), fy)
            midx = (fx + mx) / 2
            parent_bottom = min(bottom_of(f)[1], bottom_of(m)[1])
            junction_y = parent_bottom - 0.28
            vline(ax, midx, parent_bottom, junction_y)
            child_xs = sorted(pos[c][0] for c in child_ids)
            if len(child_xs) > 1:
                hline(ax, child_xs[0], child_xs[-1], junction_y)
            for c in child_ids:
                cx, cy = pos[c]
                vline(ax, cx, junction_y, top_of(c)[1])
        else:
            parent = f if f in pos else m
            if parent in pos:
                px, py = pos[parent]
                parent_bottom = bottom_of(parent)[1]
                junction_y = parent_bottom - 0.28
                vline(ax, px, parent_bottom, junction_y)
                child_xs = sorted(pos[c][0] for c in child_ids)
                if len(child_xs) > 1:
                    hline(ax, child_xs[0], child_xs[-1], junction_y)
                for c in child_ids:
                    cx, cy = pos[c]
                    vline(ax, cx, junction_y, top_of(c)[1])

    for r in persons.itertuples(index=False):
        if r.person_id in pos:
            x, y = pos[r.person_id]
            draw_person_box(ax, x, y, r.display_name)

    ax.text((min(xs) + max(xs)) / 2, max(ys) + 0.55, "家系図", ha="center", va="center",
            fontproperties=FONT_PROP, fontsize=18)

    output = BytesIO()
    plt.tight_layout()
    plt.savefig(output, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    output.seek(0)
    return output

@app.route("/")
def home():
    sample = """person_id,display_name,sex,father_id,mother_id,spouse_group_id,generation,relative_type,branch_side,birth_order,is_bloodline,is_display
P0001,須藤 彰,male,P0002,P0003,M0006,0,self,self,3,TRUE,TRUE
P0002,須藤 太郎,male,P0004,P0005,M0001,-1,father,paternal,1,TRUE,TRUE
P0003,花子,female,P0006,P0007,M0001,-1,mother,maternal,2,TRUE,TRUE
P0004,須藤 一郎,male,,,M0002,-2,paternal_grandfather,paternal,1,TRUE,TRUE
P0005,ミネ,female,,,M0002,-2,paternal_grandmother,paternal,2,TRUE,TRUE
P0006,田中 次郎,male,,,M0004,-2,maternal_grandfather,maternal,1,TRUE,TRUE
P0007,田中 和子,female,,,M0004,-2,maternal_grandmother,maternal,2,TRUE,TRUE
P0010,洋子,female,,,M0006,0,spouse,spouse,4,FALSE,TRUE
P0020,健,male,P0001,P0010,,1,son,child,1,TRUE,TRUE
P0021,美咲,female,P0001,P0010,,1,daughter,child,2,TRUE,TRUE"""
    return f"""
    <html>
    <head><meta charset="utf-8"><title>家系図作成</title></head>
    <body>
        <h2>家系図作成</h2>
        <p>persons形式のCSVを貼り付けて「作成」を押してください。</p>
        <form action="/generate" method="post">
            <textarea name="csv" rows="18" cols="90">{sample}</textarea><br>
            <button type="submit">作成</button>
        </form>
    </body>
    </html>
    """

@app.route("/generate", methods=["POST"])
def generate():
    csv_data = request.form["csv"]
    try:
        persons = read_persons_csv_text(csv_data)
        errors = validate_persons(persons)
        if errors:
            return "<br>".join(["入力エラー:"] + errors), 400
        output = create_family_tree(persons)
        return send_file(output, download_name="kakeizu.png", mimetype="image/png", as_attachment=True)
    except Exception as e:
        return f"処理中にエラーが発生しました: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
