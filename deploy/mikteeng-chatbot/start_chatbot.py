"""Compatibility launcher for existing Render settings, using root RSPM only."""
import sys
from pathlib import Path
from mikteeng_rspm.cli import main
if __name__=='__main__':
    checkpoint=Path(__file__).resolve().parents[2]/'models/synthetic_vector_demo.json'
    sys.argv=['mikteeng-rspm','serve','--checkpoint',str(checkpoint),*sys.argv[1:]]
    main()
