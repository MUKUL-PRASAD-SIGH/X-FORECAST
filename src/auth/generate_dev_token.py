"""
Generate a development JWT token for a company/user.

Usage (Windows cmd):
    py -3 src\auth\generate_dev_token.py --company tech_corp_001 --user admin@techcorp --role admin --expiry-days 7

The script reads `JWT_SECRET` from the environment or falls back to a local default `dev-secret` (ONLY for local/dev use).
Do NOT use the fallback secret in production.

It prints the token to stdout and also prints a curl-friendly Bearer header example.
"""
import os
import argparse
import time
import jwt

DEFAULT_SECRET = "dev-secret"
ALGORITHM = "HS256"


def generate_token(company_id: str, user: str, role: str = "user", expiry_days: int = 7, secret: str = None):
    secret = secret or os.environ.get("JWT_SECRET") or DEFAULT_SECRET
    now = int(time.time())
    exp = now + expiry_days * 24 * 3600
    payload = {
        "sub": user,
        "company_id": company_id,
        "role": role,
        "iat": now,
        "exp": exp
    }
    token = jwt.encode(payload, secret, algorithm=ALGORITHM)
    return token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", required=True, help="Company id (e.g. tech_corp_001)")
    parser.add_argument("--user", required=True, help="User email or id")
    parser.add_argument("--role", default="user", help="Role (user/admin)")
    parser.add_argument("--expiry-days", type=int, default=7, help="Token lifetime in days")
    parser.add_argument("--secret", default=None, help="Secret to sign the token (overrides env JWT_SECRET)")
    args = parser.parse_args()

    token = generate_token(args.company, args.user, args.role, args.expiry_days, args.secret)
    print("\n=== Development JWT Token ===\n")
    print(token)
    print("\nUse as HTTP header:\n")
    print(f"Authorization: Bearer {token}\n")
    print("Note: This script uses a fallback secret 'dev-secret' if JWT_SECRET is not set. Do not use fallback in production.")

if __name__ == '__main__':
    main()
