# Split frontend/backend deployment

Goal: the public gets `https://SERVER_NAME/` on the frontend server; the
backend server is never reachable except from that frontend server.

```
 browser  --HTTPS-->  [frontend server: nginx]  --HTTP + shared secret-->  [backend server: uvicorn]
                       serves index.html                                   FastAPI, Neo4j, Ollama
                       proxies /ask, /documents
```

The browser only ever talks to the frontend server, so there's no CORS to
configure. The backend trusts the frontend via two independent layers —
use both, don't rely on either alone:

1. **Network ACL** (does the actual enforcing) — the backend's firewall
   only accepts connections from the frontend server's IP.
2. **Shared secret header** (defense-in-depth) — nginx attaches
   `X-Internal-Proxy-Secret`; the backend rejects requests without it.
   Covers you if the firewall rule is ever fat-fingered, but by itself
   it's not sufficient — anyone who can reach the port at all can read
   this header value out of the nginx config too if that host is
   compromised, so it's a backstop, not the primary control.

## Backend server

1. `cp network_config/backend.env.example backend.env`, fill in
   `LEGALRAG_PROXY_SECRET` (`openssl rand -hex 32`) and the rest of your
   existing config, then load it before starting uvicorn (e.g.
   `set -a; source backend.env; set +a`).
2. Bind uvicorn to the private interface only if you have one, e.g.
   `--host 10.0.0.5` instead of `--host 0.0.0.0`. If there's no private
   network between the two hosts and it has to bind `0.0.0.0`, the
   firewall rule in step 3 is what's actually keeping it locked down.
3. Restrict inbound traffic to the frontend server's IP. Example with
   `ufw`:
   ```bash
   ufw default deny incoming
   ufw allow from <FRONTEND_SERVER_IP> to any port 8000
   ufw allow ssh   # don't lock yourself out
   ufw enable
   ```
   On a cloud provider, prefer doing this in the security group /
   firewall rule set instead of (or in addition to) `ufw` — e.g. AWS
   security group inbound rule scoped to the frontend's IP/CIDR, or a
   DigitalOcean/GCP VPC so the backend has no public IP at all.
4. Start the backend as usual: `python -m uvicorn backend.web:app --host <bind-addr> --port 8000`.

## Frontend server

1. Copy `frontend/index.html` to `/var/www/legalrag/index.html`.
2. Copy `network_config/nginx-frontend.conf` to
   `/etc/nginx/sites-available/legalrag.conf`, symlink into
   `sites-enabled/`, and fill in the placeholders:
   - `SERVER_NAME` — your public domain
   - `BACKEND_HOST` — the backend's private IP (preferred) or public IP
   - `PROXY_SECRET` — the same value as `LEGALRAG_PROXY_SECRET` above
3. `certbot --nginx -d SERVER_NAME` to provision TLS (do this before
   going live — don't serve a public demo over plain HTTP).
4. `nginx -t && systemctl reload nginx`.

## Verify

- `curl -i https://SERVER_NAME/ask -X POST -H 'Content-Type: application/json' -d '{"question":"test"}'`
  from your own machine should work (goes through the proxy).
- `curl -i http://BACKEND_HOST:8000/ask -X POST ...` run directly against
  the backend from anywhere other than the frontend server should be
  refused at the network level (connection refused/timeout), not just
  get a 403 — if it gets a 403 instead of timing out, the firewall rule
  isn't actually restricting the port and you're relying on the shared
  secret alone.
