"""Why did g0 call a plain parquet file malformed?

g0 refuses symlinks, non-regular files and anything with st_nlink > 1. The last
one is the trap: on the Windows drive mounted into WSL (DrvFs), a perfectly
ordinary file can report more than one link, and the gate cannot tell that from
a real hard link sharing bytes outside the submission tree.

If that is what happened, the agent is fine and the *test environment* is wrong:
local gate checks have to run on a native Linux filesystem.
"""

import os
import pathlib
import stat
import sys

for target in sys.argv[1:] or [
        "/mnt/c/Users/Saifuddin/Documents/agenthon-t1/track2_agent/out"]:
    directory = pathlib.Path(target)
    print(f"{directory}  (filesystem check)")
    for path in sorted(directory.iterdir()):
        st = os.lstat(path)
        print(f"  {path.name:<26} regular={stat.S_ISREG(st.st_mode)} "
              f"symlink={stat.S_ISLNK(st.st_mode)} nlink={st.st_nlink} "
              f"size={st.st_size}")
    print()
