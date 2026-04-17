import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from ui.app_entry import main


if __name__ == "__main__":
    main()
