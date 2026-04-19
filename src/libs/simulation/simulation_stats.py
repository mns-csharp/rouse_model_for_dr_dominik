"""SimulationStats — acceptance-rate tracker for each move type."""


class SimulationStats:
    def __init__(self):
        self.hinge_attempted = 0
        self.hinge_accepted = 0
        self.n_tail_attempted = 0
        self.n_tail_accepted = 0
        self.c_tail_attempted = 0
        self.c_tail_accepted = 0
        self.pivot_attempted = 0
        self.pivot_accepted = 0
        self.rescale_count = 0
        self.rescale_max_drift = 0.0
        self.rescale_last_drift = 0.0

    def record(self, move_type: str, accepted: bool):
        if move_type == 'hinge':
            self.hinge_attempted += 1
            if accepted:
                self.hinge_accepted += 1
        elif move_type == 'n_tail':
            self.n_tail_attempted += 1
            if accepted:
                self.n_tail_accepted += 1
        elif move_type == 'c_tail':
            self.c_tail_attempted += 1
            if accepted:
                self.c_tail_accepted += 1
        elif move_type == 'pivot':
            self.pivot_attempted += 1
            if accepted:
                self.pivot_accepted += 1

    def report(self) -> str:
        def rate(a, t):
            return f"{100.0 * a / t:.1f}%" if t > 0 else "N/A"
        return (f"Hinge: {rate(self.hinge_accepted, self.hinge_attempted)} "
                f"({self.hinge_accepted}/{self.hinge_attempted}), "
                f"N-tail: {rate(self.n_tail_accepted, self.n_tail_attempted)} "
                f"({self.n_tail_accepted}/{self.n_tail_attempted}), "
                f"C-tail: {rate(self.c_tail_accepted, self.c_tail_attempted)} "
                f"({self.c_tail_accepted}/{self.c_tail_attempted}), "
                f"Pivot: {rate(self.pivot_accepted, self.pivot_attempted)} "
                f"({self.pivot_accepted}/{self.pivot_attempted})")
