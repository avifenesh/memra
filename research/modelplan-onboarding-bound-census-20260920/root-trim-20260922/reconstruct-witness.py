"""Recreate only the non-loadable sparse inspection witness from retained header bytes.
This supplies no weights. The complete model tensor census must reject it.
"""
import argparse,json,pathlib
parser=argparse.ArgumentParser();parser.add_argument('header',type=pathlib.Path);parser.add_argument('pin',type=pathlib.Path);parser.add_argument('output',type=pathlib.Path);args=parser.parse_args()
pin=json.loads(args.pin.read_text());header=args.header.read_bytes()
with args.output.open('xb') as f:f.write(header);f.truncate(pin['bytes'])
