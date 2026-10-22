import glob

def read_brown3d_file(filepath):
    with open(filepath) as f:
        header = f.readline()
        assert header.startswith("ITEM: TIMESTEP"), f"Unexpected header, expected ITEM: TIMESTEP, got: {header!r}"
        timestep = int(f.readline())

        header = f.readline()
        assert header.startswith("ITEM: NUMBER OF ATOMS"), f"Unexpected header, expected ITEM: NUMBER OF ATOMS, got: {header!r}"
        atom_count = int(f.readline())

        header = f.readline()
        assert header.startswith("ITEM: BOX BOUNDS"), f"Unexpected header, expected ITEM: BOX BOUNDS, got: {header!r}"
        x_min, x_max = map(float, f.readline().split())
        y_min, y_max = map(float, f.readline().split())
        z_min, z_max = map(float, f.readline().split())

        header = f.readline()
        assert header.startswith("ITEM: ATOMS"), f"Unexpected header, expected ITEM: ATOMS, got: {header!r}"

        column_names = header.split()[2:]  # drop "ITEM:" and "ATOMS"
        column_index = {name: i for i, name in enumerate(column_names)}

        required_columns = ["id", "x", "y", "z", "vx", "vy", "vz"]
        missing = [c for c in required_columns if c not in column_index]
        if missing:
            raise ValueError(f"{filepath}: ITEM: ATOMS header is missing required column(s) {missing}: {header!r}")

        # Assumes exactly one timestep block per file, which holds for every
        # file these simulation scripts produce (a new file is opened each
        # output step). A general multi-frame LAMMPS dump file would need a
        # loop around everything above to pick up further TIMESTEP blocks.
        particles = []

        for i in range(atom_count):
            line = f.readline()
            if not line:
                raise ValueError(
                    f"{filepath}: header declared {atom_count} atoms, but file ended after {i} particle rows"
                )

            values = line.split()

            particle_id = int(values[column_index["id"]])
            x = float(values[column_index["x"]])
            y = float(values[column_index["y"]])
            z = float(values[column_index["z"]])
            vx = float(values[column_index["vx"]])
            vy = float(values[column_index["vy"]])
            vz = float(values[column_index["vz"]])

            particles.append(
                (particle_id, x, y, z, vx, vy, vz)
            )

    return {
        "timestep": timestep,
        "atom_count": atom_count,
        "box": {
            "x": (x_min, x_max),
            "y": (y_min, y_max),
            "z": (z_min, z_max)
        },
        "particles": particles
    }


if __name__ == "__main__":
    files = glob.glob("Brown3D-*.txt")
    for filepath in sorted(files):
        data = read_brown3d_file(filepath)
        print(filepath, "->", len(data["particles"]), "particles, timestep", data["timestep"])