"""Run the live photo extraction on one image: uv run python scripts/extract_demo.py <image>.

Needs ANTHROPIC_API_KEY in the environment.
"""

import json
import mimetypes
import sys

from deicm.web.extract import derive_prefill, extract_state


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: extract_demo.py <image>")
    path = sys.argv[1]
    media_type = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        state = extract_state(f.read(), media_type)
    print(json.dumps(state.model_dump(), indent=2))
    print("\nprefill:")
    print(json.dumps(derive_prefill(state), indent=2))


if __name__ == "__main__":
    main()
