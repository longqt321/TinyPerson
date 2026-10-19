def resolve_stages(model):
    """Find principal YOLO26 C3k2 stage outputs before the detection head."""
    layers = getattr(model, "model", None)
    if layers is None:
        raise ValueError("Expected YOLO26 model with backbone stages")
    first_head = next((i for i, layer in enumerate(layers) if type(layer).__name__ == "C2PSA"), None)
    if first_head is None:
        raise ValueError("Expected YOLO26 backbone C2PSA stage")
    c3 = [i for i, layer in enumerate(layers[:first_head]) if type(layer).__name__ == "C3k2"]
    if len(c3) != 4:
        raise ValueError(f"Expected four YOLO26 backbone C3k2 stages, found {len(c3)}")
    return dict(zip(("C2", "C3", "C4", "C5"), c3))
