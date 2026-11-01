from app.core.config import get_settings
from urllib.parse import urlsplit
import socket

settings = get_settings()
url = settings.neon_db
parsed = urlsplit(url)

print("Scheme:", parsed.scheme)
print("Host:", parsed.hostname)
print("Port:", parsed.port or 5432)

if not parsed.hostname:
    raise SystemExit("Invalid NEON_DB connection string")

try:
    addresses = socket.getaddrinfo(
        parsed.hostname,
        parsed.port or 5432,
        type=socket.SOCK_STREAM
    )

    print("DNS resolution successful!")
    print("Addresses:", sorted({a[4][0] for a in addresses}))

except OSError as error:
    print("DNS resolution failed:", error)