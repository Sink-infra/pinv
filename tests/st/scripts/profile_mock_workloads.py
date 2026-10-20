"""Profile Mock workloads using the common suite runner."""
import sys
from tests.common.profiling import profile_samples


def main():
    from tests.st.scripts.run_tests import main as run
    return run(["--suite", "mock", "--profile"] + sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
