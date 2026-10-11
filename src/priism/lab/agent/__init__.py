"""Remote GPU agent: runs lab jobs on a machine we cannot reach (a friend's PC behind his router).

Pull model: the agent polls a job queue in S3-compatible storage, runs one job at a time, streams its
log, status and heartbeat back, and uploads its results. Nothing connects to the machine, and a lost
connection costs nothing: the job keeps running locally and everything is synced when the network
returns.

- ``store``: the storage behind the queue (S3, or a local folder for tests);
- ``jobs``: job specs, the allowlist that turns a spec into a command, and the client side (submit,
  status, log, cancel, pull);
- ``agent``: the loop on the GPU machine (``priism-lab agent``).
"""
