class Event:
    def __init__(self, d: dict):
        self.dur = d.get("dur", 0)
        self.name = d.get("name")
        self.prop = d.get("prop")
        self.descend = d.get("descend", True)
        self.maxrep = d.get("maxrep", -1)

    def __repr__(self):
        disp = f"{self.name}={self.dur}"
        if self.prop:
            disp = f"{self.prop}*{disp}"
        return disp
