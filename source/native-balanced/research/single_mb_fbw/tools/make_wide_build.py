"""Generate a separate source/build variant for the 64-expert completion mask."""
from pathlib import Path

root=Path(__file__).resolve().parents[1]
text=(root/'tools/build_priority.sh').read_text()
text=text.replace('source/native-priority','source/native-wide')
text=text.replace('native-priority-source.tar','native-wide-source.tar')
text=text.replace('matched-priority','wide-priority')
text=text.replace('-DMEGAFLUX_GRADIENT_PRIORITY="$priority"','-DMEGAFLUX_EARLY_DW1_WIDE=1 -DMEGAFLUX_GRADIENT_PRIORITY="$priority"')
text=text.replace('for priority in 0 1; do',
    'tar -xf "$task/artifacts/catlass-d7d0bc2-include.tar" -C "$root/3rdparty/catlass"\nfor priority in 0 1; do')
(root/'tools/build_wide.sh').write_text(text,newline='\n')
print('WIDE_BUILD_SCRIPT_READY')
