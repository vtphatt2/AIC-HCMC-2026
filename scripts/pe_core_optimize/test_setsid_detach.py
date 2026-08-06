"""Stage 0: does `setsid` actually free the SSH connection, unlike bare
`nohup` (which held the connection for full job duration all last session,
because nohup only ignores SIGHUP -- it does NOT detach from the session)?

Not meant to be uploaded/run as a python script itself -- this file documents
the exact shell commands to run manually, since the test IS the shell/ssh
behavior, not python logic. Kept here so the commands aren't retyped from
scratch when Colab is back.

Step 1 (launch, time how long the ssh command itself takes to return):
  time ssh colab-gpu "setsid bash -c 'for i in 1 2 3 4 5 6; do echo tick \$i \$(date +%T); sleep 5; done' < /dev/null > /content/setsid_test.log 2>&1 & echo LAUNCHED; sleep 1"

  Expected if setsid works: returns in ~1-2s (not ~30s for the full 6-tick loop).
  Expected if setsid does NOT help (same as nohup): returns only after ~30s.

Step 2 (immediately after, while the job should still be running remotely):
  ssh colab-gpu "cat /content/setsid_test.log"

  Expected if truly detached: a SECOND ssh connection succeeds (no "already-active
  SSH session" 429 error) and shows partial tick output, growing on repeat calls.
"""
