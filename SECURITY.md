# Security policy

## Reporting a vulnerability

Report it privately through GitHub's private vulnerability reporting: on the repository page, open
**Security**, then **Report a vulnerability**. The report becomes a draft security advisory that only
you and the maintainer can see. Please do not open a public issue for anything exploitable.

You will get an answer within a week. Useful to include: the commit or version, how the service was
started (container or `tsguard serve`, and the `--host` it bound), and the smallest request or file
that reproduces the problem.

## Supported versions

Until 1.0 only the newest minor version receives fixes.

| Version | Supported |
| --- | --- |
| 0.1.x | yes |

## What the project defends against, and what it does not

**Defended:**

- **Malformed and oversized requests.** The request schema is strict (no type coercion, no unknown
  fields, no NaN or infinity), bounds every channel value to a magnitude of 1e6, `first_index` to
  `2**53 - 10000` and `history` to 10,000 rows, and a validation error never echoes the input back.
- **A damaged or swapped bundle file.** A bundle is checked against its sha256 sidecar before
  anything in it is unpickled, and then against digests of its own state and a fingerprint of the
  feature code that built it.
- **A damaged or substituted dataset.** Every SMD file is checked against the size and sha256
  pinned in the repository before it is parsed.

**Not defended:**

- **A bundle written by someone else.** The sidecar proves integrity, not origin: whoever can write
  the bundle can write its sidecar, and unpickling a bundle runs code. Load only bundles you built,
  and keep the directory they are served from writable by you alone (the container mounts it
  read-only).
- **Unauthenticated access.** The service has no authentication and no TLS. `tsguard serve` binds
  loopback by default and `docker-compose.yml` publishes the port on `127.0.0.1` only; anything
  wider needs a gateway in front that authenticates.
- **Request floods.** There is no rate limit. Each request costs a feature computation over the rows
  it carries, so a gateway should limit request rates and body sizes.
- **Confidentiality of the data.** SMD is a public dataset, and the bundles and reports are not
  encrypted.
