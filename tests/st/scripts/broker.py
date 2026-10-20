"""Foreground child transport for the target's existing shared NPU service."""
import argparse
import json
import os
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout',type=int,default=600)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not args.command:parser.error('a child command is required')
    from npu_runtime import NpuSession
    with NpuSession(device=0,lock_timeout=1800) as session:
        print(json.dumps({'npu_service_pid':session.service_pid,'lock_wait_seconds':session.lock_wait_seconds}),flush=True)
        result=session.run(args.command,cwd=os.getcwd(),environment=os.environ.copy(),timeout=args.timeout)
        print(result.stdout,end='',flush=True)
        print(result.stderr,end='',file=sys.stderr,flush=True)
        return result.returncode


if __name__=='__main__':raise SystemExit(main())
