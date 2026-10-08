from solver.graph import pick_belt

SPLITTER = "Build_ConveyorAttachmentSplitter_C"
MERGER = "Build_ConveyorAttachmentMerger_C"


def block_entities(block, prefix, x0, y0):
    """Defines entities following the set schema for building"""
    entities = []
    for m, (x, y) in enumerate(block["machines"]):
        entities.append({"id": f"{prefix}_m{m}", "kind": "machine", "class": block["machine"],
                         "pos": [x0 + x, y0 + y, 0.0], "yaw": 0,
                         "recipe": block["recipe"], "clock": block["clock"]})

    for s in block["splitters"]:
        c, r = s["cell"]
        entities.append({"id": f"{prefix}_s{s['input']}_{s['machine']}", 
                         "kind": "attachment", "class": SPLITTER, "yaw": 0,
                         "pos" : [x0 + c + 0.5, y0 + r + 0.5, 1.0]})

    for m in block["mergers"]:
        c, r = m["cell"]
        entities.append({"id": f"{prefix}_g{m['machine']}", 
                        "kind": "attachment", "class": MERGER, "yaw": 0,
                        "pos" : [x0 + c + 0.5, y0 + r + 0.5, 1.0]})    
    return entities

def belt_class(rate):
    """Finds the class for a belt based on its tier"""
    tier, _ = pick_belt(rate)
    return f"Build_ConveyorBeltMk{tier}_C"


def belt(cls, from_id, from_port, to_id, to_port):
    """Creates a single belt entity"""
    return {"id": f"belt_{from_id}_{to_id}", "kind": "belt", "class": cls,
            "from": from_id, "from_port": from_port, "to": to_id, "to_port": to_port}


def block_belts(block, prefix):
    n = block["n"]
    entities = []

    # belts from splitter to amchine
    for s in block["splitters"]:
        s_id = f"{prefix}_s{s['input']}_{s['machine']}"
        m_id = f"{prefix}_m{s['machine']}"
        input_cls = belt_class(block["inputs"][s["input"]]["rate"])
        entities.append(belt(input_cls, s_id, "Output2", m_id, s["port"]))

    # belts from each splitter to the next
    for j, row in enumerate(block["inputs"]):
        input_cls = belt_class(row["rate"])
        for m in range(n - 1):
            from_id = f"{prefix}_s{j}_{m}"
            to_id = f"{prefix}_s{j}_{m + 1}"
            entities.append(belt(input_cls, from_id, "Output1",
                                 to_id, "Input1"))

    # belts from machines to mergers
    output_cls = belt_class(block["output"]["rate"])
    for g in block["mergers"]:
        m = g["machine"]
        from_id = f"{prefix}_m{m}"
        to_id = f"{prefix}_g{m}"
        entities.append(belt(output_cls, from_id, g["port"], to_id, "Input3"))

    # belts from mergers to the next merger    
    for m in range(n - 1):
        entities.append(belt(output_cls, f"{prefix}_g{m}", "Output1", f"{prefix}_g{m + 1}", "Input1"))

    return entities