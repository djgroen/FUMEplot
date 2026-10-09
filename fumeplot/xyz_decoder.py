import glob

def read_xyz_file(filepath: str) -> dict:
    """
    Parse an RW1D/RW2D-style XYZ output file.

    Returns a dict with a single "particles" key, holding a list of
    (particle_id, x, y, z) tuples. No timestep or box metadata is
    returned, since the XYZ format doesn't contain either.

    Note: real/standard XYZ allows any free-text comment on line 2
    (not necessarily blank) and element-symbol-first rows ("C x y z").
    This parser matches what this project's generator actually writes:
    a blank line 2 and particle-ID-first rows ("1 x y z"). Line 2 is
    now accepted and discarded as a comment rather than required to
    be blank; rows are still assumed ID-first.
    """
    with open(filepath) as f:
        particle_count = int(f.readline())

        f.readline()  # comment line - discarded, not required to be blank

        particles = []

        for line in f:
            if not line.strip():
                continue  # skip stray blank lines, e.g. a trailing one at EOF

            parts = line.split()

            particle_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

            particles.append((particle_id, x, y, z))

        if len(particles) != particle_count:
            raise ValueError(
                f"{filepath}: header declared {particle_count} particles, found {len(particles)}"
            )

    return {"particles": particles}


if __name__ == "__main__":
    files = glob.glob("RW1D-*.txt")
    for filepath in files:
        data = read_xyz_file(filepath)
        print(filepath, "->", len(data["particles"]), "particles")

    files2d = glob.glob("RW2D-*.txt")
    for filepath in files2d:
        data = read_xyz_file(filepath)
        print(filepath, "->", len(data["particles"]), "particles")