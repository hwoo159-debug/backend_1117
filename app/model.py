# Placeholder model. We'll replace with your teammate's notebook later.
def predict(prompt: str) -> str:
    lines = [ln for ln in prompt.splitlines() if ln.strip()]
    return lines[-1].replace("Answer:", "").strip()
