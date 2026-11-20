#!/usr/bin/env bash
# The checkpoint gate's red arms: the runner with a wrong program the gate must refuse.
exec "${HY4F_RUN:?HY4F_RUN names the runner binary}" "$@" --red "${HY4F_RED:?HY4F_RED names the arm}"
