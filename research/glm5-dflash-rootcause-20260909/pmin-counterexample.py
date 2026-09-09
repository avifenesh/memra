#!/usr/bin/env python3
"""Exact enumeration at slot 2, conditioned on a previously accepted slot 1."""
from fractions import Fraction as F
import json
p=[F(1,2),F(1,2)];q=[F(9,10),F(1,10)];threshold=F(7,10)
def enumerate_rule(rule):
 out=[F(0),F(0)]
 residual=[max(F(0),pi-qi) for pi,qi in zip(p,q)];z=sum(residual)
 for token,qi in enumerate(q):
  keep=qi>=threshold if rule=='selected_q' else max(q)>=threshold
  if not keep:
   for i,pi in enumerate(p):out[i]+=qi*pi
   continue
  accept=min(F(1),p[token]/qi)
  out[token]+=qi*accept
  for i,ri in enumerate(residual):out[i]+=qi*(1-accept)*ri/z
 return [str(v) for v in out]
print(json.dumps({'target':[str(x) for x in p],'proposal':[str(x) for x in q],'pmin':str(threshold),'legacy_selected_q':enumerate_rule('selected_q'),'causal_max_q':enumerate_rule('max_q'),'condition':'slot 2 after accepted slot 1, PMIN0 off'},indent=2))
assert enumerate_rule('selected_q')==['11/20','9/20']
assert enumerate_rule('max_q')==['1/2','1/2']
